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
import torchvision.models as models

import load_data
import net

file_dir = os.getcwd()

img_transform = transform.Compose([
    transform.Resize((300,300)),
    transform.ToTensor(),
    transform.Normalize(mean = [0.485, 0.456, 0.406],
                        std = [0.229, 0.224, 0.225])
])

network = torch.load('./model/model90.2.pkl')
print(network)
test_data = load_data.Load_testdata(transform=img_transform)
test_load = Data.DataLoader(dataset=test_data, batch_size=1, shuffle=False)
labels = load_data.get_labels()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
network.to(device)
print(device)

def test_model():
    ans = []
    ids = []
    network.eval()
    with torch.no_grad():
        for data in test_load:
            x, y = data
            x = x.to(device)
            outputs = network(x)
            predict = torch.max(outputs.data, 1)[1]
            ids.extend(y)
            ans.extend(predict)
    return ans, ids

with open('answer.csv', 'w', newline='') as csvFile:
    ans, ids = test_model()
    writer = csv.writer(csvFile)
    writer.writerow(['id', 'label'])
    for i in range(len(ans)):
        writer.writerow([ids[i], labels[ans[i]]])