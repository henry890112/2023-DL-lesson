import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import dataloader
import matplotlib.pyplot as plt


class EEGNet(nn.Module):
    def __init__(self, mode):
        super(EEGNet, self).__init__()
        if mode == "ELU":
            activate_func = nn.ELU(alpha=1.0, inplace=True)
        elif mode == "ReLU":
            activate_func = nn.ReLU(inplace=True)
        elif mode == "LeakyReLU":
            activate_func = nn.LeakyReLU(negative_slope=0.01, inplace=True)

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, (1, 51), stride=(1, 1), padding=(0, 25), bias=False),
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, (2, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            activate_func,
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0),
            nn.Dropout(p=0.25)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, (1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            activate_func,
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0),
            nn.Dropout(p=0.25)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=736, out_features=2, bias=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

    def train(self, train_data, train_label, test_data, test_label, epoch=150, batch_size=64, learning_rate=1e-02, mode = "ELU"):
        # loss function
        loss_func = nn.CrossEntropyLoss()
        # optimizer
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # establish the train and test loss list
        self.train_list = []
        self.test_list = []

        # if mode == "ELU":
        #     self.ELU_train_list = []
        #     self.ELU_test_list = []
        # elif mode == "ReLU":
        #     self.ReLU_train_list = []
        #     self.ReLU_test_list = []
        # elif mode == "LeakyReLU":
        #     self.LeakyReLU_train_list = []
        #     self.LeakyReLU_test_list = []
        
        # train
        for i in range(epoch):
            # shuffle
            permutation = torch.randperm(train_data.size()[0])
            train_data = train_data[permutation]
            train_label = train_label[permutation]
            # train
            for j in range(0, train_data.size()[0], batch_size):
                # get data
                data = train_data[j:j + batch_size]
                label = train_label[j:j + batch_size]
                # forward
                out = self.forward(data)
                loss = loss_func(out, label)
                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # can change the number of epochs to visualize the accuracy
            if i % 1 == 0:
                # calculate the train accuracy
                # out = self.forward(data)  上面有計算過了不要再計算一次
                train_pred_y = torch.max(out, 1)[1].data.squeeze()
                train_accuracy = sum(train_pred_y == label) / label.size(0) * 100
                # print('Epoch: ', i, '| train loss: %.4f' % loss.data.cpu().numpy(), '| train accuracy: %.2f' % train_accuracy)

                # calculate the test accuracy
                test_out = self.forward(test_data)
                pred_y = torch.max(test_out, 1)[1].data.squeeze()
                test_accuracy = sum(pred_y == test_label) / test_label.size(0) * 100

                # print('train accuracy: %.2f' % train_accuracy, '| test accuracy: %.2f' % test_accuracy)

                # convert the torch tensor to numpy
                train_accuracy = train_accuracy.cpu().numpy()
                test_accuracy = test_accuracy.cpu().numpy()
                self.train_list.append(train_accuracy)
                self.test_list.append(test_accuracy)

    def get_train_and_test_list(self):
        return self.train_list, self.test_list
    
    # visualize the accuracy to the number of epochs and use different colors to represent the training and test accuracy
    def visualize(self, ELU_train_list, ELU_test_list, ReLU_train_list, ReLU_test_list, LeakyReLU_train_list, LeakyReLU_test_list):
        
        # use the original epoch number
        plt.plot(ELU_train_list, color='blue', label='elu_train')
        plt.plot(ELU_test_list, color='red', label='elu_test')
        plt.plot(ReLU_train_list, color='green', label='relu_train')
        plt.plot(ReLU_test_list, color='yellow', label='relu_test')
        plt.plot(LeakyReLU_train_list, color='black', label='leakyrelu_train')
        plt.plot(LeakyReLU_test_list, color='pink', label='leakyrelu_test')

        # add the title and labels
        plt.title('Activate function comparision(EEGNet)')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy(%)')
        plt.legend()
        plt.show()
        # save the figure
        plt.savefig('EEGNet.png')
        

# main function
if __name__ == '__main__':
    # import data
    train_data, train_label, test_data, test_label = dataloader.read_bci_data()

    # convert numpy to 'cuda' torch tensor
    train_data = torch.from_numpy(train_data).float().cuda()
    train_label = torch.from_numpy(train_label).long().cuda()
    test_data = torch.from_numpy(test_data).float().cuda()
    test_label = torch.from_numpy(test_label).long().cuda()


    # use gpu if available
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Running on the GPU")
        print(torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("Running on the CPU")
    
    # establish the list to store the accuracy
    ELU_train_list = []
    ELU_test_list = []
    ReLU_train_list = []
    ReLU_test_list = []
    LeakyReLU_train_list = []
    LeakyReLU_test_list = []

    # ELU
    model = EEGNet(mode = "ELU")
    model.to(device)
    model.train(train_data, train_label, test_data, test_label, mode = "ELU")
    ELU_train_list, ELU_test_list = model.get_train_and_test_list()
    # find the best test accuracy
    max_test_accuracy = max(ELU_test_list)
    print("Best ELU accuracy : {:.2f} %".format(max_test_accuracy))
    print("ELU finish")

    # ReLU
    model = EEGNet(mode = "ReLU")
    model.to(device)
    model.train(train_data, train_label, test_data, test_label, mode = "ReLU")
    ReLU_train_list, ReLU_test_list = model.get_train_and_test_list()
    # find the best test accuracy
    max_test_accuracy = max(ReLU_test_list)
    print("Best ReLU accuracy : {:.2f} %".format(max_test_accuracy))
    print("ReLU finish")

    # LeakyReLU
    model = EEGNet(mode = "LeakyReLU")
    model.to(device)
    model.train(train_data, train_label, test_data, test_label, mode = "LeakyReLU")
    LeakyReLU_train_list, LeakyReLU_test_list = model.get_train_and_test_list()
    # find the best test accuracy
    max_test_accuracy = max(LeakyReLU_test_list)
    print("Best LeakyReLU accuracy : {:.2f} %".format(max_test_accuracy))
    print("LeakyReLU finish")

    # visualize the accuracy and save the figure
    model.visualize(ELU_train_list, ELU_test_list, ReLU_train_list, ReLU_test_list, LeakyReLU_train_list, LeakyReLU_test_list)
    print("Finish all")

    # save the weights
    # torch.save(model.state_dict(), 'EEGnet.pkl')

