import pandas as pd
from torch.utils import data
import numpy as np
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import os
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def getData(mode):
    if mode == 'train':
        img = pd.read_csv('train_img.csv')
        label = pd.read_csv('train_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)
    else:
        img = pd.read_csv('test_img.csv')
        label = pd.read_csv('test_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)


class RetinopathyLoader(data.Dataset):
    def __init__(self, root, mode):
        """
        Args:
            root (string): Root path of the dataset.
            mode : Indicate procedure status(training or testing)

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.root = root
        self.img_name, self.label = getData(mode)
        self.mode = mode
        print("> Found %d images..." % (len(self.img_name)))

        # crop the image in center
        self.center_crop = transforms.CenterCrop(512)
        self.to_tensor = transforms.ToTensor() # transforms.ToTensor()，它将输入数据转换为张量形式。这个操作会将像素值从0-255的范围映射到0-1之间的浮点数，并且交换图像的通道顺序。这使得输入数据可以被传递给PyTorch的神经网络模型进行训练。

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):
        """something you should implement here"""

        """
           step1. Get the image path from 'self.img_name' and load it.
                  hint : path = root + self.img_name[index] + '.jpeg'
           
           step2. Get the ground truth label from self.label
                     
           step3. Transform the .jpeg rgb images during the training phase, such as resizing, random flipping, 
                  rotation, cropping, normalization etc. But at the beginning, I suggest you follow the hints. 
                       
                  In the testing phase, if you have a normalization process during the training phase, you only need 
                  to normalize the data. 
                  
                  hints : Convert the pixel value to [0, 1]
                          Transpose the image shape from [H, W, C] to [C, H, W]
                         
            step4. Return processed image and label
        """
        # step1:get the image path use os
        path = os.path.join(self.root, self.img_name[index] + '.jpeg')

        # step2:get the ground truth label from  self.label
        label = self.label[index]

        # step3:transform the .jpeg rgb images
        img = Image.open(path)
        
        min_size = img.size[0] if img.size[0]<img.size[1] else img.size[1] 
        transforms_pre = transforms.Compose([transforms.CenterCrop(min_size), 
                                             transforms.Resize((512, 512)),
                                             transforms.RandomHorizontalFlip(p = 0.5),
                                             transforms.RandomVerticalFlip(p = 0.5),
                                             transforms.RandomRotation(degrees = 10), 
                                             transforms.ToTensor()])
        img = transforms_pre(img)

        # img = self.center_crop(img)
        # img = self.to_tensor(img)
        # print(img.shape)        
        return img, label



# print the np.squeeze(img.values), np.squeeze(label.values)
img, label = getData('train')
print(img.shape, label.shape)

img = pd.read_csv('train_img.csv')
label = pd.read_csv('train_label.csv')
print(np.squeeze(img.values))
print(np.squeeze(label.values))