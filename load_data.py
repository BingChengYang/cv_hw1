import csv
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch
from os import listdir
from os.path import isfile, join

file_dir = os.getcwd()

# get labels_list
def get_labels():
    labels = []
    with open(file_dir+'/training_labels.csv') as csvfile:
            rows = csv.reader(csvfile)
            for row in rows:
                if row[0] == 'id':
                    continue
                if row[1] not in labels:
                    labels.append(row[1])
                if len(labels) == 196:
                    break
    return labels

def load_img(path):
    return Image.open(path).convert('RGB')

class Load_traindata(Dataset):
    def __init__(self, transform=None, target_transform=None, loader=load_img, valid=False, valid_len=1000):
        # initial variable
        imgs = []
        self.valid_imgs = []
        self.train_imgs = []
        labels = []
        # open training_label.csv
        with open(file_dir+'/training_labels.csv') as csvfile:
            rows = csv.reader(csvfile)
            for row in rows:
                if row[0] == 'id':
                    continue
                if row[1] not in labels:
                    labels.append(row[1])
                imgs.append((row[0], labels.index(row[1])))
        # divide data into training set and validation set
        train_len = len(imgs) - valid_len
        self.train_imgs = imgs[0:train_len]
        self.valid_imgs = imgs[-(valid_len+1):-1]
        self.transform = transform
        self.target_transform = target_transform
        self.loader = load_img
        self.valid = valid

    def __getitem__(self, index):
        if self.valid == False:
            filename, label = self.train_imgs[index]
        else:
            filename, label = self.valid_imgs[index]
        img = self.loader(file_dir+'/training_data/training_data/'+filename+'.jpg')
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        if self.valid == False:
            return len(self.train_imgs)
        else:
            return len(self.valid_imgs)

class Load_testdata(Dataset):
    def __init__(self, transform=None, target_transform=None, loader=load_img):
        # read the data in testing_data folder
        self.imgs = [f for f in listdir(file_dir+'/testing_data/testing_data/') if isfile(join(file_dir+'/testing_data/testing_data/', f))]
        self.transform = transform
        self.target_transform = target_transform
        self.loader = load_img

    def __getitem__(self, index):
        filename = self.imgs[index]
        img = self.loader(file_dir+'/testing_data/testing_data/'+filename)
        if self.transform is not None:
            img = self.transform(img)
        return img, filename.split('.')[0]

    def __len__(self):
        return len(self.imgs)