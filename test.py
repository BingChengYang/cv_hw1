import os
from os import listdir
from os.path import isfile, join
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.utils.data as Data
import torchvision.transforms as transform
import torch
import torch.nn as nn
import numpy as np
import csv

import load_data
import net

file_dir = os.getcwd()

img_transform = transform.Compose([
    transform.Resize((128,128)),
    transform.ToTensor(),
    transform.Normalize(mean = [0.5],
                        std = [0.229])
])

network = torch.load('model.pkl')
test_data = load_data.Load_testdata(transform=img_transform)
test_load = Data.DataLoader(dataset=test_data, batch_size=1, shuffle=False)
labels = load_data.get_labels()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
network.to(device)
print(device)

def test_model():
    ans = []
    with torch.no_grad():
        for data in test_load:
            x = data
            x = x.to(device)
            outputs = network(x)
            predict = torch.max(outputs.data, 1)[1]
            ans.append(labels[predict])
    return ans

with open('answer.csv', 'w', newline='') as csvFile:
    ans = test_model()
    ids = ids = [f.split('.')[0] for f in listdir(file_dir+'/testing_data/testing_data/') if isfile(join(file_dir+'/testing_data/testing_data/', f))]
    writer = csv.writer(csvFile)
    writer.writerow(['id', 'label'])
    for i in range(len(ans)):
        writer.writerow([ids[i], ans[i]])