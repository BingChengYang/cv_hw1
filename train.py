import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transform

import torchvision.models as models

# my own library
import load_data
import net

# test model
def test_model(load):
    correct = 0
    total = 0
    network.eval()
    with torch.no_grad():
        for data in load:
            x, y = data
            x,y = x.to(device), y.to(device)
            outputs = network(x)
            predict = torch.max(outputs.data, 1)[1]
            total += y.size(0)
            correct += (predict==y).sum().item()
    print("accuracy : {:f} %".format(float(correct)/float(total) * 100.0))
    return float(correct)/float(total) * 100.0

# setting hyperparameter
learning_rate = 0.00001
epoches = 10
mini_batch_size = 32
load_model= True
img_size = 300

train_img_transform = transform.Compose([
    # transform.Resize((img_size, img_size)),
    # # transform.ColorJitter(saturation=0.5),
    # transform.RandomCrop(img_size),
    # transform.RandomHorizontalFlip(),
    transform.RandomResizedCrop(img_size),
    transform.RandomHorizontalFlip(),
    transform.ToTensor(),
    transform.Normalize(mean = [0.485, 0.456, 0.406],
                        std = [0.229, 0.224, 0.225])
])

valid_img_transform = transform.Compose([
    transform.Resize((img_size, img_size)),
    transform.ToTensor(),
    transform.Normalize(mean = [0.485, 0.456, 0.406],
                        std = [0.229, 0.224, 0.225])
])

if load_model == True:
    # network = models.resnet50(pretrained=True)
    # fc_features = network.fc.in_features
    # network.fc = nn.Linear(fc_features, 196)
    network = torch.load('./model/model.pkl')
else:
    network = net.CNN_test()
print(network)
train_set = load_data.Load_traindata(transform=train_img_transform)
val_set = load_data.Load_traindata(transform=valid_img_transform, valid=True)
train_load = Data.DataLoader(dataset=train_set, batch_size=mini_batch_size, shuffle=True)
val_load = Data.DataLoader(dataset=val_set, batch_size=mini_batch_size, shuffle=True)

optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate, weight_decay=0.0001)
loss_func = nn.CrossEntropyLoss()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
network.to(device)

# Start training model
for epoch in range(epoches):
    print(epoch)
    network.train()
    for step, (x, y) in enumerate(train_load):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = network(x)
        loss = loss_func(output, y)
        loss.backward()
        optimizer.step()
    # print(torch.max(output.data, 1)[1])
    print(loss)
    val_acc = test_model(val_load)
    torch.save(network, './model/model' +str(val_acc) + '.pkl')
    if epoch%5 == 0:
        test_model(train_load)
test_model(train_load)
test_model(val_load)
# save model
torch.save(network, './model/model.pkl')