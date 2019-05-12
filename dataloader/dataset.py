import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset
from PIL import  Image
EXTENSIONS = ['.jpg', '.png','.JPG','.PNG']
import random

import torch.nn as nn
import torch.utils.data as data
import torch
import cv2
import os
from PIL import Image
import numpy as np
import random
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize,Normalize
import matplotlib.pyplot as plt

import os
from os.path import join
from os import listdir
class SRdata(data.Dataset):
    def __init__(self, args):
        self.args=args
        root_dir='./data/dataset'
        hr_dir='img'
        lr_dir='gtFine'

        self.file_HR=os.path.join(root_dir,hr_dir)
        # self.file_LR = os.path.join(root_dir, lr_dir)
        self.image_filenames = [join(self.file_HR, x) for x in listdir( self.file_HR) ]
        # self.image_filenames = [join(self.file_HR, x) for x in listdir(self.file_HR)]

        self.len=len(self.image_filenames)

    def get_patch(self,img_path, img_seg_path):
        img_hr = cv2.imread(img_path)     #h  w  c
        img_seg = Image.open(img_seg_path)
        img_seg = np.array(img_seg)  # 宽×高×


        h, w =img_seg.shape
        min =w
        if h<w:
            min =h
        if min<200:
            img_seg = Image.open(img_seg_path)
            img_seg = img_seg.resize((200, 200), Image.BILINEAR)
            img_seg = np.array(img_seg)  # 宽×高×
        # 裁剪得到图片的400x400大小图片
            img_hr = cv2.imread(img_path)
            img_hr = cv2.resize(img_hr, (200, 200), interpolation=cv2.INTER_CUBIC)
            return img_hr, img_seg
        h_random =random.randint(0,h-200)
        w_random=random.randint(0,w-200)
        img_seg = img_seg[h_random:h_random + 200, w_random:w_random + 200]
        img_hr =img_hr[h_random:h_random+200,w_random:w_random+200,:]    #.transpose((2,0,1))

        return img_hr, img_seg
    def __getitem__(self, index):
        img_path = self.image_filenames[index]

             #宽×高×通道
        # img_hr = cv2.resize(img_hr, (200, 200), interpolation=cv2.INTER_CUBIC) #宽×高×通道
        try:
            img_seg_path = './data/dataset/profiles/'+img_path.split('\\')[-1].split('.')[0] + '-profile.jpg'

            img_hr, img_seg =self.get_patch(img_path,img_seg_path)
            # img_seg = img_seg.resize((200, 200), Image.BILINEAR)

            #裁剪得到图片的400x400大小图片
            # img_seg = cv2.resize(img_seg, (200, 200), interpolation=cv2.INTER_CUBIC)


            img_seg[img_seg <= 50] = 1
            img_seg[img_seg > 50] = 0
            out_hr=img_hr / 256.0 - 0.5
            out_hr = out_hr.transpose((2,0,1))
        except:
            print(img_path)
        return out_hr, img_seg       #3   400  400    /400   400

    def __len__(self):
        return self.len





def load_image(file):
    return Image.open(file)

def is_image(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS)

def image_path(root, basename, extension):
    return os.path.join(root, '{}{}'.format(basename,extension))

def image_path_city(root, name):
    return os.path.join(root, '{}'.format(name))

def image_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])

class NeoData(Dataset):
    def __init__(self, imagepath=None, labelpath=None, transform=None):
        #  make sure label match with image 
        self.transform = transform 
        assert os.path.exists(imagepath), "{} not exists !".format(imagepath)
        assert os.path.exists(labelpath), "{} not exists !".format(labelpath)                                  
        self.image = []
        self.label= [] 
        with open(imagepath,'r') as f:
            for line in f:
                self.image.append(line.strip())
        with open(labelpath,'r') as f:
            for line in f:
                self.label.append(line.strip())

    def __getitem__(self, index):
        filename = self.image[index]
        filenameGt = self.label[index]
        
        with open(filename, 'rb') as f: 
            image = load_image(f).convert('RGB')
        with open(filenameGt, 'rb') as f:
            label = load_image(f).convert('P')
        if self.transform is not None:
            image, label = self.transform(image, label)
        return image, label

    def __len__(self):
        return len(self.image)
    
class NeoData_test(Dataset):
    def __init__(self, imagepath=None, labelpath=None, transform=None):
        self.transform = transform 
        
        assert os.path.exists(imagepath), "{} not exists !".format(imagepath)
        assert os.path.exists(labelpath), "{} not exists !".format(labelpath)
        
        self.image = []
        self.label= [] 
        with open(imagepath,'r') as f:
            for line in f:
                self.image.append(line.strip())
        with open(labelpath,'r') as f:
            for line in f:
                self.label.append(line.strip())
        print("Length of test data is {}".format(len(self.image)))
    def __getitem__(self, index):
        filename = self.image[index]
        filenameGt = self.label[index]
        
        with open(filename, 'rb') as f: # advance
            image = load_image(f).convert('RGB')
        with open(filenameGt, 'rb') as f:
            label = load_image(f).convert('P')

        if self.transform is not None:
            image_tensor, label_tensor, img = self.transform(image, label)

        return (image_tensor, label_tensor, np.array(img))  #return original image, in order to show segmented area in origin

    def __len__(self):
        return len(self.image)

