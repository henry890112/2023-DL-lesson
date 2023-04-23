import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import RetinopathyLoader
from torch.utils.data import DataLoader
from torchsummary import summary
from torchvision import transforms, datasets, models
import argparse
from tqdm import tqdm
from torchvision.models import ResNet50_Weights, ResNet18_Weights

class Basicblock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, channel, stride=1):
        super(Basicblock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channel)
        self.conv2 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channel)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channel != self.expansion*channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, self.expansion*channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*channel)
            )

    def forward(self, x):
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = nn.ReLU()(out)
        return out
    
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, channel, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, channel, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channel)
        self.conv2 = nn.Conv2d(channel, channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channel)
        self.conv3 = nn.Conv2d(channel, self.expansion*channel, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*channel)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channel != self.expansion*channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, self.expansion*channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*channel)
            )

    def forward(self, x):
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        out = nn.ReLU()(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = nn.ReLU()(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=4):
        super(ResNet, self).__init__()
        self.in_channel = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.classify = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(in_features=512 * block.expansion, out_features=50),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.25),
                nn.Linear(in_features=50, out_features=5))
        # self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, channel, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channel, channel, stride))
            self.in_channel = channel * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.classify(out)
        return out

def ResNet18():
    return ResNet(Basicblock, [2,2,2,2])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

# train_acc_list = []
# test_acc_list = []
def new_train(name, model, device, optimizer, EPOCH):
    train_acc_list = []
    test_acc_list = []
    for epoch in range(EPOCH):
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        model.train()
        loop = tqdm(train_loader, total=len(train_loader), leave=True)
        train_acc = 0

        for data, target in loop:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model.forward(data)  
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            optimizer.step()
            loop.set_description('Epoch: {}, Loss: {:.4f}'.format(epoch+1, loss.item()))
            _, train_pred = torch.max(output,1)
            train_acc += (train_pred == target).sum().item()
            # print(train_acc)

        # 每個epoch的訓練結果
        train_acc_list.append(train_acc/len(train_loader.dataset)*100)
        print(train_acc_list)
        torch.save(model.state_dict(), './model/{}_{}.pth'.format(name, epoch+1))
        print("{}_model_saved_{}.pth".format(name, epoch+1))
        test_acc = evaluate(model, test_loader)  # record the accuracy of test dataset, need to divided by 7026
        test_acc_list.append(test_acc/len(test_loader.dataset)*100)
        print(test_acc_list)

    return train_acc_list, test_acc_list

def evaluate(model, Loader):
    test_acc = 0
    model.to(device)
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(Loader)):
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device).long()
            output = model(inputs)
            # print("output: {}".format(output))
            _,Predicted=torch.max(output,1)
            # print("labels: {}".format(labels))
            test_acc+=(Predicted==labels).sum().item()
    # results = accuracy_score(labels, result)
    return test_acc


def train(name, model, device, optimizer, EPOCH):
    train_acc_list = []
    for epoch in range(EPOCH):
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        model.train()
        loop = tqdm(train_loader, total=len(train_loader), leave=True)
        train_acc = 0

        for data, target in loop:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model.forward(data)  
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            optimizer.step()
            loop.set_description('Epoch: {}, Loss: {:.4f}'.format(epoch+1, loss.item()))
            _, train_pred = torch.max(output,1)
            train_acc += (train_pred == target).sum().item()
            # print(train_acc)
        torch.save(model, './model/{}_{}.pth'.format(name, epoch+1))
        print("{}_model_saved{}.pth".format(name, epoch+1))
        # 每個epoch的訓練結果
        train_acc_list.append(train_acc/len(train_loader.dataset)*100)
        print(train_acc_list)
        torch.cuda.empty_cache()
    return train_acc_list

def test(name, model, device, EPOCH):
    test_acc_list = [] 
    for epoch in range(EPOCH):
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        model.load_state_dict(torch.load('./model/{}_{}.pth'.format(name, epoch+1)))
        model.to(device)
        model.eval()   # 不會更新參數
        loop = tqdm(test_loader, total=len(test_loader), leave=True)
        test_acc = 0
        for data, target in loop:
            data, target = data.to(device), target.to(device)
            output = model.forward(data)
            _, test_pred = torch.max(output,1)
            test_acc += (test_pred == target).sum().item()
        test_acc_list.append(test_acc/len(test_loader.dataset)*100)
        print(test_acc_list)
        torch.cuda.empty_cache()
    return test_acc_list
    

def plt_result(pretrain_acc_list, pretest_acc_list, train_acc_list, test_acc_list, name):
    import matplotlib.pyplot as plt
    import numpy as np
    x = np.arange(1, 6, 1)
    plt.plot(x, pretrain_acc_list, label='with pretrain_acc')
    plt.plot(x, pretest_acc_list, label='with pretest_acc')
    plt.plot(x, train_acc_list, label='w/o pretrain_acc')
    plt.plot(x, test_acc_list, label='w/o pretest_acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy(%)')
    plt.title('Accuracy of {} model'.format(name))
    plt.legend()
    plt.show()
    # save the figure
    plt.savefig('Accuracy.png')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def makeconfusionmatrix(model, name):
    y_pred = []
    y_true = []

    # load the model weights
    model.load_state_dict(torch.load('./model/{}.pth'.format(name)))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    model.to(device)
    model.eval()

    with torch.no_grad():
        for i, (images, target) in enumerate(test_loader):
            images=images.to(device)
            target=target.to(device)
            output = model.forward(images)
            _, preds = torch.max(output, 1) 
            y_pred.extend(preds.view(-1).detach().cpu().numpy())       
            y_true.extend(target.view(-1).detach().cpu().numpy())
            print(i,'/',len(test_loader))
    cf_matrix_normalize=confusion_matrix(y_true,y_pred,normalize='true')
    return cf_matrix_normalize

import pandas as pd
import seaborn as sns
def plot_confusion_matrix(cf_matrix,name):
    class_names = ['no DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
    df_cm = pd.DataFrame(cf_matrix, class_names, class_names) 
    sns.heatmap(df_cm, annot=True, cmap='Oranges')
    plt.title(name)
    plt.xlabel("prediction")
    plt.ylabel("laTbel (ground truth)")
    plt.show()
    # save the figure
    plt.savefig('confusion_matrix.png')


if __name__ == '__main__':

    # use argparse to parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default="resnet18", help='use train or pretrained model (default: train)')  
    args = parser.parse_args()
    # check the gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Running on the GPU")
        print(torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("Running on the CPU")

    

    # load the data in data_loader.py
    train_dataset = RetinopathyLoader(root = "./data/train_resize" ,mode = "train")
    test_dataset = RetinopathyLoader(root = "./data/test_resize", mode = "test")

    if args.mode == "resnet18": 
        EPOCH = 10
        BATCH_SIZE = 16
        LR = 0.01
        name = "res18_pt"
        res18_pt=models.resnet18(pretrained = True)
        res18_pt.fc=nn.Linear(512,5)
        res18_pt.to(device)
        optimizer = optim.SGD(res18_pt.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        # pretrain_acc_list = train(name, res18_pt, device, optimizer, EPOCH)
        # pretest_acc_list = test(name, res18_pt, device, EPOCH)
        pretrain_acc_list, pretest_acc_list = new_train(name, res18_pt, device, optimizer, EPOCH)

        name = "res18_npt"
        res18_npt=models.resnet18(pretrained = False)
        res18_npt.fc=nn.Linear(512,5)
        res18_npt.to(device)
        optimizer = optim.SGD(res18_npt.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        # train_acc_list = train(name, res18_npt, device, optimizer, EPOCH)
        # test_acc_list = test(name, res18_npt, device, EPOCH)
        train_acc_list, test_acc_list = new_train(name, res18_npt, device, optimizer, EPOCH)
        
        # get the highest accuracy in pretrain and train 
        print("max pretrain acc:{} in epoch:{} ".format(max(pretrain_acc_list), pretrain_acc_list.index(max(pretrain_acc_list))+1))
        print("max train acc:{} in epoch:{} ".format(max(train_acc_list), train_acc_list.index(max(train_acc_list))+1))
        print("max pretest acc:{} in epoch:{} ".format(max(pretest_acc_list), pretest_acc_list.index(max(pretest_acc_list))+1))
        print("max test acc:{} in epoch:{} ".format(max(test_acc_list), test_acc_list.index(max(test_acc_list))+1)) 

        # plot the result
        name = "ResNet18"
        plt_result(pretrain_acc_list, pretest_acc_list, train_acc_list, test_acc_list, name)

        
    elif args.mode == "resnet50":
        EPOCH = 5
        BATCH_SIZE = 8
        LR = 0.01
        name = "res50_pt"
        res50_pt=models.resnet50(pretrained = True)
        res50_pt.fc=nn.Linear(2048,5)
        res50_pt.to(device)
        optimizer = optim.SGD(res50_pt.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        # pretrain_acc_list = train(name, res50_pt, device, optimizer, EPOCH)
        # pretest_acc_list = test(name, res50_pt, device, EPOCH)
        pretrain_acc_list, pretest_acc_list = new_train(name, res50_pt, device, optimizer, EPOCH)

        name = "res50_npt"
        res50_npt=models.resnet50(pretrained = False)
        res50_npt.fc=nn.Linear(2048,5)
        res50_npt.to(device)
        optimizer = optim.SGD(res50_npt.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        # train_acc_list = train(name, res50_npt, device, optimizer, EPOCH)
        # test_acc_list = test(name, res50_npt, device, EPOCH)
        train_acc_list, test_acc_list = new_train(name, res50_npt, device, optimizer, EPOCH)

        # get the highest accuracy in pretrain and train 
        print("max pretrain acc:{} in epoch:{} ".format(max(pretrain_acc_list), pretrain_acc_list.index(max(pretrain_acc_list))+1))
        print("max train acc:{} in epoch:{} ".format(max(train_acc_list), train_acc_list.index(max(train_acc_list))+1))
        print("max pretest acc:{} in epoch:{} ".format(max(pretest_acc_list), pretest_acc_list.index(max(pretest_acc_list))+1))
        print("max test acc:{} in epoch:{} ".format(max(test_acc_list), test_acc_list.index(max(test_acc_list))+1))

        # plot the result
        name = "ResNet50"
        plt_result(pretrain_acc_list, pretest_acc_list, train_acc_list, test_acc_list, name)


    elif args.mode == "confusion_matrix":
        BATCH_SIZE = 16
        # res18_pt=models.resnet18(pretrained=True)
        # res18_pt.fc=nn.Linear(512,5)
        res50_pt=models.resnet50(pretrained=True)
        res50_pt.fc=nn.Linear(2048,5)
        name = "res50_pt_5"
        cf_matrix_normalize = makeconfusionmatrix(res50_pt, name)
        plot_confusion_matrix(cf_matrix_normalize,'resnet50_pretrained')


