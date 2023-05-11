import argparse
import itertools
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import bair_robot_pushing_dataset
from models.lstm import gaussian_lstm, lstm
from models.vgg_64 import vgg_decoder, vgg_encoder
from utils import init_weights, kl_criterion, plot_pred, finn_eval_seq, pred, mse_criterion

torch.backends.cudnn.benchmark = True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.002, type=float, help='learning rate')
    parser.add_argument('--beta1', default=0.9, type=float, help='momentum term for adam')
    parser.add_argument('--batch_size', default=4, type=int, help='batch size')
    parser.add_argument('--log_dir', default='./logs/fp', help='base directory to save logs')
    parser.add_argument('--model_dir', default='', help='base directory to save logs')
    parser.add_argument('--data_root', default='./data', help='root directory for data')
    parser.add_argument('--optimizer', default='adam', help='optimizer to train with')
    parser.add_argument('--niter', type=int, default=300, help='number of epochs to train for')
    parser.add_argument('--epoch_size', type=int, default=600, help='epoch size')
    parser.add_argument('--tfr', type=float, default=1.0, help='teacher forcing ratio (0 ~ 1)')
    parser.add_argument('--tfr_start_decay_epoch', type=int, default=120, help='The epoch that teacher forcing ratio become decreasing')
    parser.add_argument('--tfr_decay_step', type=float, default=1/150, help='The decay step size of teacher forcing ratio (0 ~ 1)')
    parser.add_argument('--tfr_lower_bound', type=float, default=0, help='The lower bound of teacher forcing ratio for scheduling teacher forcing ratio (0 ~ 1)')
    parser.add_argument('--kl_anneal_cyclical', default=True, action='store_true', help='use cyclical mode')
    parser.add_argument('--kl_anneal_ratio', type=float, default=0.5, help='The decay ratio of kl annealing')
    parser.add_argument('--kl_anneal_cycle', type=int, default=3, help='The number of cycle for kl annealing during training (if use cyclical mode)')
    parser.add_argument('--seed', default=1, type=int, help='manual seed')
    parser.add_argument('--n_past', type=int, default=2, help='number of frames to condition on')
    parser.add_argument('--n_future', type=int, default=10, help='number of frames to predict')
    parser.add_argument('--n_eval', type=int, default=30, help='number of frames to predict at eval time')
    parser.add_argument('--rnn_size', type=int, default=256, help='dimensionality of hidden layer')
    parser.add_argument('--posterior_rnn_layers', type=int, default=1, help='number of layers')
    parser.add_argument('--predictor_rnn_layers', type=int, default=2, help='number of layers')
    parser.add_argument('--z_dim', type=int, default=64, help='dimensionality of z_t')
    parser.add_argument('--g_dim', type=int, default=128, help='dimensionality of encoder output vector and decoder input vector')
    parser.add_argument('--beta', type=float, default=0.0001, help='weighting on KL to prior')
    parser.add_argument('--num_workers', type=int, default=4, help='number of data loading threads')
    parser.add_argument('--last_frame_skip', action='store_true', help='if true, skip connections go between frame t and frame t+t rather than last ground truth frame')
    parser.add_argument('--cuda', default=True, action='store_true')  
    # add the train and test mode
    parser.add_argument('--mode', default='train', help='train or test')

    args = parser.parse_args()
    return args
# def the training process to use in the main
def train(x, cond, modules, optimizer, kl_anneal, args, device):
    torch.cuda.empty_cache()
    modules['frame_predictor'].zero_grad()
    modules['posterior'].zero_grad()
    modules['encoder'].zero_grad()
    modules['decoder'].zero_grad()

    '''
    posterior 是指条件概率分布 $P(z|x_{1:t})$，即给定过去的观测序列 $x_{1:t}$，
    计算当前时刻的隐状态 $z_t$ 的概率分布。在这个模型中，posterior 是由神经网络模型实现的，
    用于将过去的观测序列编码成当前时刻的隐状态。具体地，它将过去的观测序列 $x_{1:t}$ 作为输入，
    经过编码操作后，输出当前时刻的隐状态 $z_t$ 的分布 $P(z_t|x_{1:t})$。
    '''

    # initialize the hidden state.
    modules['frame_predictor'].hidden = modules['frame_predictor'].init_hidden()
    modules['posterior'].hidden = modules['posterior'].init_hidden()
    mse = 0
    kld = 0

    # when random[0, 1)大於tfr, 就不用teacher forcing
    use_teacher_forcing = True if random.random() < args.tfr else False

    # TODO training the cvae and use the teacher forcing(也就是groud truth) or not

   
    h_seq = [modules['encoder'](x[i]) for i in range(args.n_past + args.n_future)]
    # define the loss funcction
    # mse_criterion = nn.MSELoss()

    # 創建一個list去存取所預測出來的x, 且要先將第一個x append進去才可以train
    x_pred_seq = []
    x_pred_seq.append(x[0])

    for i in range(1, args.n_past + args.n_future):
        # 下面所寫的 h_target = h_t ; h = h_t-1
        if use_teacher_forcing == True:
            ######1######
            # 當前frame的h
            h_target = h_seq[i][0]   # 第0維是h, 第1維是skip
            # 前一個frame的h
            if args.last_frame_skip or i < args.n_past:	
                h, skip = h_seq[i-1]
            else:
                h = h_seq[i-1][0] # teacher forcing use ground truth
            # 得到當前frame的z_t() 並計算其kl loss
            # posterior计算当前时刻的隐状态 z_t 的概率分布
            z_t, mu, logvar = modules['posterior'](h_target)
            ######2######
            # 而這裡是用z_t和h_pred來預測下一個frame
            # 利用自己先前所預測的frame來預測下一個frame，增加robustness
            # if use_teacher_forcing == False:就會使用ground truth去做訓練
            

            # 因為是cvae所以也要將x[i-1]的condition加入
            # print(cond[i-1].shape)  # torch.Size([12, 7])
            # only frame predictor的那個lstm要增加維度
            h_pred = modules['frame_predictor'](torch.cat([cond[i-1], h, z_t], 1))
            x_pred = modules['decoder']([h_pred, skip])
            x_pred_seq.append(x_pred)
        else:
            ######1######
            # 當前frame的h
            h_target = h_seq[i][0]   # 第0維是h, 第1維是skip
            # 前一個frame的h
            if args.last_frame_skip or i < args.n_past:	
                h, skip = h_seq[i-1]
            else:
                h, _ = modules['encoder'](x_pred_seq[i-1]) # #non teacher forcing use predict frame as input
            # 得到當前frame的z_t() 並計算其kl loss
            # posterior计算当前时刻的隐状态 z_t 的概率分布
            z_t, mu, logvar = modules['posterior'](h_target)
            ######2######
            # 而這裡是用z_t和h_pred來預測下一個frame
            # 利用自己先前所預測的frame來預測下一個frame，增加robustness
            # if use_teacher_forcing == False:就會使用ground truth去做訓練
        
            # 因為是cvae所以也要將x[i-1]的condition加入
            # print(cond[i-1].shape)  # torch.Size([12, 7])
            # only frame predictor的那個lstm要增加維度
            h_pred = modules['frame_predictor'](torch.cat([ cond[i-1], h, z_t], 1))
            x_pred = modules['decoder']([h_pred, skip])
            x_pred_seq.append(x_pred)
        #  利用預測的x_pred和真實的x[i]來計算mse
        mse += mse_criterion(x_pred, x[i])
        kld += kl_criterion(mu, logvar, args)


    beta = kl_anneal.get_beta()
    # change beta to tensor
    # beta = torch.tensor(beta)
    # reconstruction loss + KL loss
    loss = mse + kld * beta


    loss.backward()

    optimizer.step()
    # return loss, mse, kld的平均值 （n_past+n_future是論文中的z_{t+1}的維度, 也就是z_dim)
    return loss.detach().cpu().numpy() / (args.n_past + args.n_future), mse.detach().cpu().numpy() / (args.n_past + args.n_future), kld.detach().cpu().numpy() / (args.n_future + args.n_past)


# annealing皆是用來解決gradient vanishing的問題
class kl_annealing():
    # TODO KL annealing
    def __init__(self, args):
        super().__init__()

        # 因為kl更新次數的關係所以要兩個相乘
        iter = args.niter * args.epoch_size

        # 先將所有的loss_weight設為1
        self.L = np.ones(iter)

        if (args.kl_anneal_cyclical==True):
            cycle = args.kl_anneal_cycle   # default = 3
        else:
            cycle = 1   

        period = iter/cycle
        ratio  = args.kl_anneal_ratio  # default = 0.5
        step = 1.0/(period*ratio)  # linear schedule 計算斜率的意思

        # 此處就先把所有的beta算好存在一個list中
        # 用此for迴圈將不同cycle中的同一個beta值都算好
        for c in range(cycle):  
            v, i = 0, 0
            while v <= 1 and (int(i+c*period) < iter):  #cycle* period = iteration
                self.L[int(i+c*period)] = v
                v += step  # plus the slope
                i += 1   # 計算iteration的次數

        self.beta = 0
        
    def update(self):
        self.beta += 1
    
    def get_beta(self):
        beta = self.L[self.beta]
        self.update()

        return beta
    
def main():
    args = parse_args()
    if args.cuda:
        assert torch.cuda.is_available(), 'CUDA is not available.'
        device = 'cuda'
    else:
        device = 'cpu'
    
    assert args.n_past + args.n_future <= 30 and args.n_eval <= 30
    assert 0 <= args.tfr and args.tfr <= 1
    assert 0 <= args.tfr_start_decay_epoch 
    assert 0 <= args.tfr_decay_step and args.tfr_decay_step <= 1

    if args.model_dir != '':
        # load model and continue training from checkpoint
        saved_model = torch.load('%s/model.pth' % args.model_dir)
        optimizer = args.optimizer
        model_dir = args.model_dir
        niter = args.niter
        args = saved_model['args']
        args.optimizer = optimizer
        args.model_dir = model_dir
        args.log_dir = '%s/continued' % args.log_dir
        start_epoch = saved_model['last_epoch']
        print("Load the model success")
    else:
        name = 'rnn_size=%d-predictor-posterior-rnn_layers=%d-%d-n_past=%d-n_future=%d-lr=%.4f-g_dim=%d-z_dim=%d-last_frame_skip=%s-beta=%.7f'\
            % (args.rnn_size, args.predictor_rnn_layers, args.posterior_rnn_layers, args.n_past, args.n_future, args.lr, args.g_dim, args.z_dim, args.last_frame_skip, args.beta)

        args.log_dir = '%s/%s' % (args.log_dir, name)
        niter = args.niter
        start_epoch = 0

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs('%s/gen/' % args.log_dir, exist_ok=True)
    os.makedirs('%s/epoch_weight/' % args.log_dir, exist_ok=True)

    print("Random Seed: ", args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if os.path.exists('./{}/train_record.txt'.format(args.log_dir)):
        os.remove('./{}/train_record.txt'.format(args.log_dir))
    
    print(args)

    with open('./{}/train_record.txt'.format(args.log_dir), 'a') as train_record:
        train_record.write('args: {}\n'.format(args))

    # ------------ build the models  --------------

    if args.model_dir != '':
        frame_predictor = saved_model['frame_predictor']
        posterior = saved_model['posterior']
    else:
        # TODO change the dim of input (add the condition)
        frame_predictor = lstm(args.g_dim+args.z_dim+7, args.g_dim, args.rnn_size, args.predictor_rnn_layers, args.batch_size, device)
        posterior = gaussian_lstm(args.g_dim, args.z_dim, args.rnn_size, args.posterior_rnn_layers, args.batch_size, device)
        frame_predictor.apply(init_weights)
        posterior.apply(init_weights)
            
    if args.model_dir != '':
        decoder = saved_model['decoder']
        encoder = saved_model['encoder']
    else:
        encoder = vgg_encoder(args.g_dim)
        decoder = vgg_decoder(args.g_dim)
        encoder.apply(init_weights)
        decoder.apply(init_weights)
    
    # --------- transfer to device ------------------------------------
    frame_predictor.to(device)
    posterior.to(device)
    encoder.to(device)
    decoder.to(device)

    # --------- load a dataset ------------------------------------
    train_data = bair_robot_pushing_dataset(args, 'train')
    validate_data = bair_robot_pushing_dataset(args, 'validate')
    test_data = bair_robot_pushing_dataset(args, 'test')

    train_loader = DataLoader(train_data,
                            num_workers=args.num_workers,
                            batch_size=args.batch_size,
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True)
    train_iterator = iter(train_loader)

    validate_loader = DataLoader(validate_data,
                            num_workers=args.num_workers,
                            batch_size=args.batch_size,
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True)
    validate_iterator = iter(validate_loader)

    test_loader = DataLoader(test_data,
                            num_workers=args.num_workers,
                            batch_size=args.batch_size,
                            shuffle=False)
    test_iterator = iter(test_loader)



    # ---------------- optimizers ----------------
    if args.optimizer == 'adam':
        args.optimizer = optim.Adam
    elif args.optimizer == 'rmsprop':
        args.optimizer = optim.RMSprop
    elif args.optimizer == 'sgd':
        args.optimizer = optim.SGD
    else:
        raise ValueError('Unknown optimizer: %s' % args.optimizer)

    params = list(frame_predictor.parameters()) + list(posterior.parameters()) + list(encoder.parameters()) + list(decoder.parameters())
    optimizer = args.optimizer(params, lr=args.lr, betas=(args.beta1, 0.999))
    kl_anneal = kl_annealing(args)

    # 將modeuls存成字典
    modules = {
        'frame_predictor': frame_predictor,
        'posterior': posterior,
        'encoder': encoder,
        'decoder': decoder,
    }
    # --------- training loop ------------------------------------

    progress = tqdm(total=args.niter)
    best_val_psnr = 0
    for epoch in range(start_epoch, start_epoch + niter):
        torch.cuda.empty_cache()
        frame_predictor.train()
        posterior.train()
        encoder.train()   # .train() 就是跑forward
        decoder.train()

        epoch_loss = 0
        epoch_mse = 0
        epoch_kld = 0

        for _ in range(args.epoch_size):
            try:
                seq, cond = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                seq, cond = next(train_iterator)
    
            seq = seq.to(device)
            cond = cond.to(device)
            seq = seq.permute(1, 0, 2, 3 ,4)
            cond = cond.permute(1, 0, 2)
            loss, mse, kld = train(seq, cond, modules, optimizer, kl_anneal, args, device)
            epoch_loss += loss
            epoch_mse += mse
            epoch_kld += kld
        
        if epoch >= args.tfr_start_decay_epoch:
            # TODO Update teacher forcing ratio 
            args.tfr -= args.tfr_decay_step
            if args.tfr < 0:
                args.tfr = 0

        progress.update(1)
        with open('./{}/train_record.txt'.format(args.log_dir), 'a') as train_record:
            train_record.write(('[epoch: %02d] loss: %.5f | mse loss: %.5f | kld loss: %.5f | teacher ratio: %.5f | KL ratio: %.5f\n' % (epoch, epoch_loss  / args.epoch_size, epoch_mse / args.epoch_size, epoch_kld / args.epoch_size, args.tfr, kl_anneal.get_beta())))
        # 存取epoch_weight
        if (epoch%10==0):
            torch.save({
                        'encoder': encoder,
                        'decoder': decoder,
                        'frame_predictor': frame_predictor,
                        'posterior': posterior,
                        'args': args,
                        'last_epoch': epoch},
                        '%s/epoch_weight/model_epoch%s.pth' % (args.log_dir,str(epoch)))

        frame_predictor.eval()
        encoder.eval()
        decoder.eval()
        posterior.eval()

        if args.mode == 'train':
            # TODO train
            if epoch % 5 == 0:
                psnr_list = []
                for _ in range(len(validate_data) // args.batch_size):
                    try:
                        validate_seq, validate_cond = next(validate_iterator)
                    except StopIteration:
                        validate_iterator = iter(validate_loader)
                        validate_seq, validate_cond = next(validate_iterator)

                    validate_seq = validate_seq.permute(1, 0, 2, 3, 4).to(device)
                    validate_cond = validate_cond.permute(1, 0, 2).to(device)
                    pred_seq = pred(validate_seq, validate_cond, modules, args, device)
                    _, _, psnr = finn_eval_seq(validate_seq[args.n_past:args.n_past + args.n_future], pred_seq[args.n_past:args.n_past + args.n_future])
                    psnr_list.append(psnr)
                    
                ave_psnr = np.mean(np.concatenate(psnr))


                with open('./{}/train_record.txt'.format(args.log_dir), 'a') as train_record:
                    train_record.write(('====================== validate psnr = {:.5f} ========================\n'.format(ave_psnr)))
        
                if ave_psnr > best_val_psnr:
                    best_val_psnr = ave_psnr
                    # save the model
                    torch.save({
                        'encoder': encoder,
                        'decoder': decoder,
                        'frame_predictor': frame_predictor,
                        'posterior': posterior,
                        'args': args,
                        'last_epoch': epoch},
                        '%s/model.pth' % args.log_dir)

                print("best_val_psnr: ", best_val_psnr)
                
            # TODO validation
            if epoch % 20 == 0:
                try:
                    validate_seq, validate_cond = next(validate_iterator)
                except StopIteration:
                    validate_iterator = iter(validate_loader)
                    validate_seq, validate_cond = next(validate_iterator)

                validate_seq = validate_seq.permute(1, 0, 2, 3, 4).to(device)
                validate_cond = validate_cond.permute(1, 0, 2).to(device)
                # plot_pred(validate_seq, validate_cond, modules, epoch, args, device)
        elif args.mode == 'test':
            test_psnr_list = []
            for _ in range(len(test_data) // args.batch_size):
                try:
                    test_seq, test_cond = next(test_iterator)
                except StopIteration:
                    test_iterator = iter(test_loader)
                    test_seq, test_cond = next(test_iterator)
                test_seq = test_seq.permute(1, 0, 2, 3, 4).to(device)
                test_cond = test_cond.permute(1, 0, 2).to(device)
                pred_seq = pred(test_seq, test_cond, modules, args, device)
                # plot_pred(test_seq, test_cond, modules, args)
                _, _, psnr = finn_eval_seq(test_seq[args.n_past:args.n_past+args.n_future], pred_seq[args.n_past:args.n_past+args.n_future])
                test_psnr_list.append(psnr)
                    
            ave_psnr = np.mean(np.concatenate(psnr))
            print("TEST PSNR: ", ave_psnr)
if __name__ == '__main__':
    main()
        
