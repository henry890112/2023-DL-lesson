import dataloader
from EEGnet import EEGNet
import torch
import matplotlib.pyplot as plt
import argparse



# compare the accuracy
# test EEGNet in different batch size
def test_batchsize_EEGNet(epoch = 100, batch_size = 16):
    # create the model
    model = EEGNet(mode = "ReLU")
    model.to(device)
    # train the model
    model.train(train_data, train_label, test_data, test_label, epoch=epoch, batch_size=batch_size)
    # get the train and test accuracy list
    train_list, test_list = model.get_train_and_test_list()

    return train_list, test_list

# test EEGNet in different learning rate
def test_lr_EEGNet(epoch = 100, learning_rate = 0.001):
    # create the model
    model = EEGNet(mode="ReLU")
    model.to(device)

    # train the model
    model.train(train_data, train_label, test_data, test_label, epoch=epoch, learning_rate=learning_rate)
    # get the train and test accuracy list
    train_list, test_list = model.get_train_and_test_list()

    return train_list, test_list

def test_optimizer_EEGNet(epoch = 200, batch_size = 16, learning_rate = 0.001, optimizer = "Adam"): 
    # create the model
    model = EEGNet(mode="ReLU")
    model.to(device)

    # train the model
    model.train(train_data, train_label, test_data, test_label, epoch=epoch, batch_size=batch_size, learning_rate=learning_rate, optimizer=optimizer)
    # get the train and test accuracy list
    train_list, test_list = model.get_train_and_test_list()

    return train_list, test_list

def visualize_batch(train_list_16, test_list_16, train_list_32, test_list_32, train_list_64, test_list_64, train_list_128, test_list_128, train_list_256, test_list_256, train_list_512, test_list_512, train_list_1024, test_list_1024):

    # set the size of the figure
    plt.figure(figsize=(10, 6))

    # plot the accuracy
    plt.plot(train_list_16, color='blue', label='train_batchsize_16')
    plt.plot(train_list_32, color='green', label='train_batchsize_32')
    plt.plot(train_list_64, color='black', label='train_batchsize_64')
    plt.plot(train_list_128, color='purple', label='train_batchsize_128')
    plt.plot(train_list_256, color='brown', label='train_batchsize_256')
    plt.plot(train_list_512, color='cyan', label='train_batchsize_512')
    plt.plot(train_list_1024, color='olive', label='train_batchsize_1024')

    # add the title and labels
    plt.title('batch size train comparision(EEGNet)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy(%)')
    # draw a dotted line in accuracy 80%
    plt.axhline(y=80, color='r', linestyle='--')
    plt.legend()
    plt.show()
    # save the figure
    plt.savefig('train_batchsize_EEGNet.png', dpi=500)

    # clean the figure
    plt.clf()
    # set the size of the figure
    plt.figure(figsize=(10, 6))
    # plot the accuracy
    plt.plot(test_list_16, color='red', label='test_batchsize_16')
    plt.plot(test_list_32, color='yellow', label='test_batchsize_32')
    plt.plot(test_list_64, color='pink', label='test_batchsize_64')
    plt.plot(test_list_128, color='orange', label='test_batchsize_128')
    plt.plot(test_list_256, color='gray', label='test_batchsize_256')
    plt.plot(test_list_512, color='magenta', label='test_batchsize_512')
    plt.plot(test_list_1024, color='lime', label='test_batchsize_1024')

    # add the title and labels
    plt.title('batch size test comparision(EEGNet)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy(%)')
    # draw a dotted line in accuracy 80%
    plt.axhline(y=80, color='r', linestyle='--')
    plt.legend()
    plt.show()
    # save the figure
    plt.savefig('test_batchsize_EEGNet.png', dpi=500)

def visualize_lr(train_list_1, test_list_1, train_list_2, test_list_2, train_list_3, test_list_3, train_list_4, test_list_4, train_list_5, test_list_5):

    # set the size of the figure
    plt.figure(figsize=(10, 6))

    # plot the accuracy
    plt.plot(train_list_1, color='blue', label='train_lr_1')
    plt.plot(train_list_2, color='green', label='train_lr_0.1')
    plt.plot(train_list_3, color='black', label='train_lr_0.01')
    plt.plot(train_list_4, color='purple', label='train_lr_0.001')
    plt.plot(train_list_5, color='brown', label='train_lr_0.0001')

    # add the title and labels
    plt.title('learning rate train comparision(EEGNet)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy(%)')
    # draw a dotted line in accuracy 80%
    plt.axhline(y=80, color='r', linestyle='--')
    plt.legend()
    plt.show()
    # save the figure
    plt.savefig('train_lr_EEGNet.png', dpi=500)

    # clean the figure
    plt.clf()
    # set the size of the figure
    plt.figure(figsize=(10, 6))
    # plot the accuracy
    plt.plot(test_list_1, color='red', label='test_lr_1')
    plt.plot(test_list_2, color='yellow', label='test_lr_0.1')
    plt.plot(test_list_3, color='pink', label='test_lr_0.01')
    plt.plot(test_list_4, color='orange', label='test_lr_0.001')
    plt.plot(test_list_5, color='gray', label='test_lr_0.0001')

    # add the title and labels
    plt.title('learning rate test comparision(EEGNet)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy(%)')
    # draw a dotted line in accuracy 80%
    plt.axhline(y=80, color='r', linestyle='--')
    plt.legend()
    plt.show()
    # save the figure
    plt.savefig('test_lr_EEGNet.png', dpi=500)

def visualize_optimizer(train_list_1, test_list_1, train_list_2, test_list_2, train_list_3, test_list_3):
    
        # set the size of the figure
        plt.figure(figsize=(10, 6))
    
        # plot the accuracy
        plt.plot(train_list_1, color='blue', label='train_optimizer_SGD')
        plt.plot(train_list_2, color='green', label='train_optimizer_Adam')
        plt.plot(train_list_3, color='black', label='train_optimizer_RMSprop')
    
        # add the title and labels
        plt.title('optimizer train comparision(EEGNet)')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy(%)')
        # draw a dotted line in accuracy 80%
        plt.axhline(y=80, color='r', linestyle='--')
        plt.legend()
        plt.show()
        # save the figure
        plt.savefig('train_optimizer_EEGNet.png', dpi=500)
    
        # clean the figure
        plt.clf()
        # set the size of the figure
        plt.figure(figsize=(10, 6))
        # plot the accuracy
        plt.plot(test_list_1, color='red', label='test_optimizer_SGD')
        plt.plot(test_list_2, color='yellow', label='test_optimizer_Adam')
        plt.plot(test_list_3, color='pink', label='test_optimizer_RMSprop')
    
        # add the title and labels
        plt.title('optimizer test comparision(EEGNet)')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy(%)')
        # draw a dotted line in accuracy 80%
        plt.axhline(y=80, color='r', linestyle='--')
        plt.legend()
        plt.show()
        # save the figure
        plt.savefig('test_optimizer_EEGNet.png', dpi=500)


# main function
if __name__ == '__main__':
    # import data
    train_data, train_label, test_data, test_label = dataloader.read_bci_data()

    # convert numpy to 'cuda' torch tensor
    train_data = torch.from_numpy(train_data).float().cuda()   # torch.Size([1080, 1, 2, 750])
    train_label = torch.from_numpy(train_label).long().cuda()
    test_data = torch.from_numpy(test_data).float().cuda()     # torch.Size([1080, 1, 2, 750])
    test_label = torch.from_numpy(test_label).long().cuda()

    # use gpu if available
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Running on the GPU")
        print(torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("Running on the CPU")

    # 利用parser = argparse.ArgumentParser()去選擇要執行的function
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='batch_size', help='choose the mode of other test')
    args = parser.parse_args()
    
    if args.mode == 'batch_size':
        ####test the accuracy in different batch size####
        # establish seven different list to store the accuracy
        train_list_16, test_list_16 = [], []
        train_list_32, test_list_32 = [], []
        train_list_64, test_list_64 = [], []
        train_list_128, test_list_128 = [], []
        train_list_256, test_list_256 = [], []
        train_list_512, test_list_512 = [], []
        train_list_1024, test_list_1024 = [], []

        # test batch size
        train_list_16, test_list_16 = test_batchsize_EEGNet(batch_size = 16)
        train_list_32, test_list_32 = test_batchsize_EEGNet(batch_size = 32)
        train_list_64, test_list_64 = test_batchsize_EEGNet(batch_size = 64)
        train_list_128, test_list_128 = test_batchsize_EEGNet(batch_size = 128)
        train_list_256, test_list_256 = test_batchsize_EEGNet(batch_size = 256)
        train_list_512, test_list_512 = test_batchsize_EEGNet(batch_size = 512)
        train_list_1024, test_list_1024 = test_batchsize_EEGNet(batch_size = 1024)

        # visualize the accuracy
        visualize_batch(train_list_16, test_list_16, train_list_32, test_list_32, train_list_64, test_list_64, train_list_128, test_list_128, train_list_256, test_list_256, train_list_512, test_list_512, train_list_1024, test_list_1024)
        print('batch size comparision(EEGNet) is done!')

    elif args.mode == 'lr':
        #######test the accuracy in different learning rate########
        # establish five different list to store the accuracy
        train_list_1, test_list_1 = [], []
        train_list_2, test_list_2 = [], []
        train_list_3, test_list_3 = [], []
        train_list_4, test_list_4 = [], []
        train_list_5, test_list_5 = [], []

        # test learning rate
        train_list_1, test_list_1 = test_lr_EEGNet(learning_rate = 1)
        train_list_2, test_list_2 = test_lr_EEGNet(learning_rate = 0.1)
        train_list_3, test_list_3 = test_lr_EEGNet(learning_rate = 0.01)
        train_list_4, test_list_4 = test_lr_EEGNet(learning_rate = 0.001)
        train_list_5, test_list_5 = test_lr_EEGNet(learning_rate = 0.0001)

        # visualize the accuracy
        visualize_lr(train_list_1, test_list_1, train_list_2, test_list_2, train_list_3, test_list_3, train_list_4, test_list_4, train_list_5, test_list_5)
        print('learning rate comparision(EEGNet) is done!')
    
    elif args.mode == 'optimizer':
        #######test the accuracy in different optimizer########
        # establish three different list to store the accuracy
        train_list_1, test_list_1 = [], []
        train_list_2, test_list_2 = [], []
        train_list_3, test_list_3 = [], []

        # test optimizer
        train_list_1, test_list_1 = test_optimizer_EEGNet(optimizer = 'SGD')
        train_list_2, test_list_2 = test_optimizer_EEGNet(optimizer = 'Adam')
        train_list_3, test_list_3 = test_optimizer_EEGNet(optimizer = 'RMSprop')

        # visualize the accuracy
        visualize_optimizer(train_list_1, test_list_1, train_list_2, test_list_2, train_list_3, test_list_3)
        print('optimizer comparision(EEGNet) is done!')

