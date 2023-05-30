''' 
This script does conditional image generation on MNIST, using a diffusion model

This code is modified from,
https://github.com/cloneofsimo/minDiffusion

Diffusion model is based on DDPM,
https://arxiv.org/abs/2006.11239

The conditioning idea is taken from 'Classifier-Free Diffusion Guidance',
https://arxiv.org/abs/2207.12598

This technique also features in ImageGen 'Photorealistic Text-to-Image Diffusion Modelswith Deep Language Understanding',
https://arxiv.org/abs/2205.11487

'''

from typing import Dict, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import os
import json
from PIL import Image
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from evaluator import evaluation_model

class CLEVRDataset(Dataset):
    def __init__(self,img_path,json_path):
        """
        :param img_path: file of training images
        :param json_path: train.json
        """

        self.img_path=img_path
        self.max_objects=0
        with open(os.path.join('objects.json'),'r') as file:
            self.classes = json.load(file)
        self.numclasses=len(self.classes)
        self.img_names=[]
        self.img_conditions=[]
        with open(json_path,'r') as file:
            dict=json.load(file)

            for img_name,img_condition in dict.items():
                self.img_names.append(img_name)
                self.max_objects=max(self.max_objects,len(img_condition))
                self.img_conditions.append([self.classes[condition] for condition in img_condition])
        self.transformations=transforms.Compose([transforms.Resize((64,64)),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        img=Image.open(os.path.join(self.img_path,self.img_names[index])).convert('RGB')
        img=self.transformations(img)
        condition=self.int2onehot(self.img_conditions[index])
        return img,condition

    def int2onehot(self,int_list):
        onehot=torch.zeros(self.numclasses)
        for i in int_list:
            onehot[i]=1.
        # onehot=onehot.type(torch.LongTensor)
        return onehot

    
class ResidualConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()
        '''
        standard ResNet style convolutional block
        '''
        self.same_channels = in_channels==out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            # this adds on correct residual in case channels have increased
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2 
            return out / 1.414
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2


class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()
        '''
        process and downscale the image feature maps
        '''
        layers = [ResidualConvBlock(in_channels, out_channels), nn.MaxPool2d(2)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()
        '''
        process and upscale the image feature maps
        '''
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x


class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        '''
        generic one layer FC NN for embedding things  
        '''
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)


class ContextUnet(nn.Module):
    def __init__(self, in_channels, n_feat = 256, n_classes=24):
        super(ContextUnet, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_classes = n_classes

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        self.down1 = UnetDown(n_feat, n_feat) 
        self.down2 = UnetDown(n_feat, 2 * n_feat)
        self.down3 = UnetDown(2 * n_feat, 4 * n_feat)

        self.to_vec = nn.Sequential(nn.AvgPool2d(8), nn.GELU())

        self.timeembed0 = EmbedFC(1, 4*n_feat)
        self.timeembed1 = EmbedFC(1, 2*n_feat)
        self.timeembed2 = EmbedFC(1, 1*n_feat)
        self.contextembed0 = EmbedFC(n_classes, 4*n_feat)
        self.contextembed1 = EmbedFC(n_classes, 2*n_feat)
        self.contextembed2 = EmbedFC(n_classes, 1*n_feat)

        self.up0 = nn.Sequential(
            # nn.ConvTranspose2d(6 * n_feat, 2 * n_feat, 7, 7), # when concat temb and cemb end up w 6*n_feat
            nn.ConvTranspose2d(4 * n_feat, 4 * n_feat, 8, 8), # otherwise just have 2*n_feat
            nn.GroupNorm(8, 4 * n_feat),
            nn.ReLU(),
        )

        self.up1 = UnetUp(8 * n_feat, 2*n_feat)   #!!!!!!!!!!!!!!!!!!!!!!!!!
        self.up2 = UnetUp(4 * n_feat, n_feat)
        self.up3 = UnetUp(2 * n_feat, n_feat)
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

    def forward(self, x, c, t, context_mask = None):
        # x is (noisy) image, c is context label, t is timestep, 
        # context_mask says which samples to block the context on

        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        hiddenvec = self.to_vec(down3)  
        # print(c, c.shape)
        # # convert context to one hot embedding
        # # c = nn.functional.one_hot(c, num_classes=self.n_classes).type(torch.float)
        # print(c.shape, context_mask.shape)

        # # mask out context if context_mask == 1
        # # context_mask = context_mask[:, None]
        # print(c.shape, context_mask.shape)

        # # context_mask = context_mask.repeat(1,self.n_classes)
        # context_mask = (-1*(1-context_mask)) # need to flip 0 <-> 1
        # # print the c and context mask shapes
        # print(c.shape, context_mask.shape)

        # c = c * context_mask
        
        # embed context, time step
        cemb1 = self.contextembed0(c).view(-1, self.n_feat * 4, 1, 1)  # torch.Size([128, 256, 1, 1])
        temb1 = self.timeembed0(t).view(-1, self.n_feat * 4, 1, 1)   # torch.Size([128, 256, 1, 1])
        cemb2 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)  # torch.Size([128, 128, 1, 1])
        temb2 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1) # torch.Size([128, 128, 1, 1])
        cemb3 = self.contextembed2(c).view(-1, self.n_feat, 1, 1) # torch.Size([128, 64, 1, 1])
        temb3 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)  # torch.Size([128, 64, 1, 1])

        # could concatenate the context embedding here instead of adaGN
        # hiddenvec = torch.cat((hiddenvec, temb1, cemb1), 1)

        up1 = self.up0(hiddenvec)   # torch.Size([128, 256, 8, 8])
        # print(up1.shape)
        up2 = self.up1(cemb1*up1+ temb1, down3)  # add and multiply embeddings  # torch.Size([128, 256, 16, 16])
        # print(cemb1.shape, up1.shape, temb1.shape, down3.shape)
        # up2 = self.up1(up1, down2) # if want to avoid add and multiply embeddings 
        # print the shape of up2
        # print(up2.shape)  # torch.Size([128, 64, 16, 16])
        up3 = self.up2(cemb2*up2+ temb2, down2)  # add and multiply embeddings  # torch.Size([128, 128, 32, 32])
        up4 = self.up3(cemb3*up3+ temb3, down1)
        out = self.out(torch.cat((up4, x), 1))
        return out


def ddpm_schedules(beta1, beta2, T):
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }


class DDPM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device, drop_prob=0.1):
        super(DDPM, self).__init__()
        self.nn_model = nn_model.to(device)

        # register_buffer allows accessing dictionary produced by ddpm_schedules
        # e.g. can access self.sqrtab later
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()

    def forward(self, x, c):
        """
        this method is used in training, so samples t and noise randomly
        """

        _ts = torch.randint(1, self.n_T+1, (x.shape[0],)).to(self.device)  # t ~ Uniform(0, n_T)
        noise = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * noise
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.

        # dropout context with some probability
        # context_mask = torch.bernoulli(torch.zeros_like(c)+self.drop_prob).to(self.device)
        
        # return MSE between added noise, and our predicted noise
        # return self.loss_mse(noise, self.nn_model(x_t, c, _ts / self.n_T, context_mask))
        return self.loss_mse(noise, self.nn_model(x_t, c, _ts / self.n_T, c))


    def sample(self, n_sample, size, c_i, device):
        # we follow the guidance sampling scheme described in 'Classifier-Free Diffusion Guidance'
        # to make the fwd passes efficient, we concat two versions of the dataset,
        # one with context_mask=0 and the other context_mask=1
        # we then mix the outputs with the guidance scale, w
        # where w>0 means more guidance
        # n_sample為照片數量

        # noise shape (照片數量, 3, 64, 64)
        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1), sample initial noise
        x_i_store = [] # keep track of generated steps in case want to plot something 
        print()
        for i in range(self.n_T, 0, -1):
            print(f'sampling timestep {i}',end='\r')
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(n_sample,1,1,1)

            # # double batch
            # x_i = x_i.repeat(2,1,1,1)
            # t_is = t_is.repeat(2,1,1,1)

            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0

            # split predictions and compute weighting
            # print the type of x_i, c_i, t_is, context_mask
            # print(x_i.dtype, len(c_i), t_is.dtype)
            eps = self.nn_model(x_i, c_i, t_is,  context_mask= None)
            # eps1 = eps[:n_sample]
            # eps2 = eps[n_sample:]
            # eps = (1+guide_w)*eps1 - guide_w*eps2
            x_i = x_i[:n_sample]
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )
            
            if i%20==0 or i==self.n_T or i<8:  
                x_i_store.append(x_i.detach().cpu().numpy())
        
        x_i_store = np.array(x_i_store)
        return x_i, x_i_store


def train():

    # hardcoding these here
    n_epoch = 60
    batch_size = 64
    n_T = 400 # 500
    device = "cuda:0"
    n_classes = 24
    n_feat = 64 # 128 ok, 256 better (but slower)
    lrate = 1e-4
    save_model = True
    save_dir = './data/'
    # ws_test = [0.0, 0.5, 2.0] # strength of generative guidance

    ddpm = DDPM(nn_model=ContextUnet(in_channels=3, n_feat=n_feat, n_classes=n_classes), betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1)
    ddpm.to(device)
    # summary(ddpm, [(3, 64, 64), (n_classes,)])
    print(ddpm)

    # optionally load a model
    ddpm.load_state_dict(torch.load("./ddpm_model.pth"))

    tf = transforms.Compose([transforms.ToTensor()]) # mnist is already normalised 0 to 1

    # dataset = MNIST("./data", train=True, download=True, transform=tf)
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=5)
    # load trainning data
    train_data = CLEVRDataset('iclevr','train.json')
    dataloader=DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    #load testig data
    print('train data size:', len(train_data), len(dataloader))  #18009 , 18009//batch_size

    with open(os.path.join('test.json'), 'r') as f:
        test_data = json.load(f)
    with open(os.path.join('new_test.json'), 'r') as f:
        new_test_data = json.load(f)
    with open(os.path.join('objects.json'), 'r') as f:
        object_data = json.load(f)

    test_cond_list = []
    for conds in test_data:
        # print(conds)
        one_hot_array = np.zeros((n_classes,), dtype=float)
        for cond in conds:
            one_hot_array[object_data[cond]] = 1
        test_cond_list.append(one_hot_array)

    new_test_cond_list = []
    for conds in new_test_data:
        # print(conds)
        one_hot_array = np.zeros((n_classes,), dtype=float)
        for cond in conds:
            one_hot_array[object_data[cond]] = 1
        new_test_cond_list.append(one_hot_array)

    optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)
    writer = SummaryWriter('./log')
    num_total = 0
    # import the evaluation model
    evaluation = evaluation_model()
    score_list = []
    best_score = 0

    for ep in range(n_epoch):
        print(f'\nepoch {ep}')
        ddpm.train()

        # linear lrate decay
        optim.param_groups[0]['lr'] = lrate*(1-ep/n_epoch)

        pbar = tqdm(dataloader)
        loss_ema = None
        for x, c in pbar:
            optim.zero_grad()
            x = x.to(device)
            c = c.to(device)
            # print float or int in x, c
            # print(x.dtype, c.dtype)
            loss = ddpm(x, c)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()

            writer.add_scalar('loss', loss.item(), num_total/141)
            num_total += 1

            
        
        # for eval, save an image of currently generated samples (top rows)
        # followed by real images (bottom rows)
        ddpm.eval()
        with torch.no_grad():
            n_sample = len(test_cond_list)
            # for w_index, w in enumerate(ws_test):
            # print(test_cond_list)
            cond_tensor = torch.FloatTensor(np.array(test_cond_list)).to(device)
            # print(cond_tensor)
            x_gen, x_gen_store = ddpm.sample(n_sample, (3, 64, 64), cond_tensor, device)
            save_image(x_gen, save_dir + f"image_ep{ep}.png", normalize=True, range=(-1, 1))

        score = evaluation.eval(x_gen, cond_tensor)
        print('\ntest_score:', score)
        score_list.append(score)
        if score>best_score:
            best_score=score
            print("best score:", best_score)
            torch.save(ddpm.state_dict(),'./ddpm_model.pth')
      
        # new_test score
        with torch.no_grad():
            n_sample = len(new_test_cond_list)
            # for w_index, w in enumerate(ws_test):
            # print(test_cond_list)
            cond_tensor_new = torch.FloatTensor(np.array(new_test_cond_list)).to(device)
            # print(cond_tensor)
            new_x_gen, x_gen_store = ddpm.sample(n_sample, (3, 64, 64), cond_tensor_new, device)
            save_image(new_x_gen, save_dir + f"new_test_image_ep{ep}.png", normalize=True, range=(-1, 1))
        score_new = evaluation.eval(new_x_gen, cond_tensor_new)
        print('\nnew_test_score:', score_new)

    # optionally save model
        if save_model and ep == int(n_epoch-1):
            torch.save(ddpm.state_dict(), save_dir + f"model_{ep}.pth")
            print('saved model at ' + save_dir + f"model_{ep}.pth")

if __name__ == "__main__":
    train()
