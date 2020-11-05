import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transform
import matplotlib
import matplotlib.pyplot as plt

# my own library
import load_data
import net

# test model
def test_model(load):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in load:
            x, y = data
            x,y = x.to(device), y.to(device)
            outputs = network(x)
            predict = torch.max(outputs.data, 1)[1]
            total += y.size(0)
            correct += (predict==y).sum().item()
    print("accuracy : {:f} %".format(float(correct)/float(total) * 100.0))

# setting hyperparameter
learning_rate = 0.001
epoches = 100
mini_batch_size = 32

img_transform = transform.Compose([
    transform.Resize((32,32)),
    transform.RandomHorizontalFlip(),
    transform.ToTensor(),
    transform.Normalize(mean = [0.485],
                        std = [0.229])
])

network = net.ResNet(net.ResidualBlock)
train_data = load_data.Load_traindata(transform=img_transform)
train_set, val_set = torch.utils.data.random_split(train_data, [len(train_data)-1500, 1500])
train_load = Data.DataLoader(dataset=train_set, batch_size=mini_batch_size, shuffle=True)
val_load = Data.DataLoader(dataset=val_set, batch_size=mini_batch_size, shuffle=True)

optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
loss_func = nn.CrossEntropyLoss()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
network.to(device)

# Start training model
for epoch in range(epoches):
    print(epoch)
    for step, (x, y) in enumerate(train_load):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = network(x)
        loss = loss_func(output, y)
        loss.backward()
        optimizer.step()
    print(loss)
    if epoch%5 == 0:
        test_model(train_load)
        test_model(val_load)
test_model(train_load)
test_model(val_load)
# save model
torch.save(network, 'model.pkl')