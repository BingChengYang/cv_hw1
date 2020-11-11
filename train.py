import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transform
import torchvision.models as models
import argparse

# my own library
import load_data
import net

# test model
def test_model(load):
    correct = 0
    total = 0
    # turn the model into evaluate mode
    network.eval()
    with torch.no_grad():
        for data in load:
            x, y = data
            x,y = x.to(device), y.to(device)
            outputs = network(x)
            predict = torch.max(outputs.data, 1)[1]
            total += y.size(0)
            correct += (predict==y).sum().item()
    print("accuracy : {:f} %".format(float(correct)/float(total)*100.0))
    return float(correct)/float(total) * 100.0

# setting hyperparameter
parser = argparse.ArgumentParser()
parser.add_argument("--model", default="model.pkl")
parser.add_argument("--lr", default=0.00001)
parser.add_argument("--epoches", default=10)
parser.add_argument("--mini_batch_size", default=32)
parser.add_argument("--load_model", default=False)
parser.add_argument("--img_size", default=300)
args = parser.parse_args()

# set the training data transform 
train_img_transform = transform.Compose([
    transform.RandomResizedCrop(args.img_size),
    transform.RandomHorizontalFlip(),
    transform.ToTensor(),
    transform.Normalize(mean = [0.485, 0.456, 0.406],
                        std = [0.229, 0.224, 0.225])
])

# set the validation data transform
valid_img_transform = transform.Compose([
    transform.Resize((args.img_size, args.img_size)),
    transform.ToTensor(),
    transform.Normalize(mean = [0.485, 0.456, 0.406],
                        std = [0.229, 0.224, 0.225])
])

# load the data and the model
if args.load_model == True:
    network = torch.load('./model/'+args.model)
else:
    network = models.resnet50(pretrained=True)
    fc_features = network.fc.in_features
    network.fc = nn.Linear(fc_features, 196)
print(network)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
network.to(device)
train_set = load_data.Load_traindata(transform=train_img_transform)
val_set = load_data.Load_traindata(transform=valid_img_transform, valid=True)
train_load = Data.DataLoader(dataset=train_set, batch_size=args.mini_batch_size, shuffle=True)
val_load = Data.DataLoader(dataset=val_set, batch_size=args.mini_batch_size, shuffle=True)

# set the loss function and the optimizer
optimizer = torch.optim.Adam(network.parameters(), lr=args.lr, weight_decay=0.0001)
loss_func = nn.CrossEntropyLoss()

# Start training model
for epoch in range(args.epoches):
    print(epoch)
    network.train()
    for step, (x, y) in enumerate(train_load):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = network(x)
        loss = loss_func(output, y)
        loss.backward()
        optimizer.step()
    print(loss)
    val_acc = test_model(val_load)
    torch.save(network, './model/model'+str(val_acc)+'.pkl')
    if (epoch % 5) == 0:
        test_model(train_load)
# save model
torch.save(network, './model/model.pkl')