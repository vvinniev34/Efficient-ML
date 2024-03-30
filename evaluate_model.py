import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR100
import torchvision.transforms as tt
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split,ConcatDataset
import pickle

clusters = [128, 256, 512, 1024, 2048, 4096, 6144, 8192, 9216]
kmeans_128 = []
kmeans_256 = []
kmeans_512 = []
kmeans_1024 = []
kmeans_2048 = []
kmeans_4096 = []
kmeans_6144 = []
kmeans_8192 = []
kmeans_9216 = []
kmeans = {
    128: kmeans_128,
    256: kmeans_256,
    512: kmeans_512,
    1024: kmeans_1024,
    2048: kmeans_2048,
    4096: kmeans_4096,
    6144: kmeans_6144,
    8192: kmeans_8192,
    9216: kmeans_9216,
}

for cluster in clusters:
    with open(f'kmeans_{cluster}.pkl', 'rb') as f:
        kmeans[cluster] = pickle.load(f)

def find_closest_centroid(row, centroids):
    distances = np.linalg.norm(centroids - row, axis=1)  # Calculate Euclidean distances
    closest_row_index = np.argmin(distances)  # Find the index of the closest row
    return centroids[closest_row_index]

stats = ((0.5074,0.4867,0.4411),(0.2011,0.1987,0.2025))
train_transform = tt.Compose([
    tt.RandomHorizontalFlip(),
    tt.RandomCrop(32,padding=4,padding_mode="reflect"),
    tt.ToTensor(),
    tt.Normalize(*stats)
])

test_transform = tt.Compose([
    tt.ToTensor(),
    tt.Normalize(*stats)
])

train_data = CIFAR100(download=True,root="./data",transform=train_transform)
test_data = CIFAR100(download=True, root="./data",train=False,transform=test_transform)

train_classes_items = dict()

for train_item in train_data:
    label = train_data.classes[train_item[1]]
    if label not in train_classes_items:
        train_classes_items[label] = 1
    else:
        train_classes_items[label] += 1

test_classes_items = dict()
for test_item in test_data:
    label = test_data.classes[test_item[1]]
    if label not in test_classes_items:
        test_classes_items[label] = 1
    else:
        test_classes_items[label] += 1

BATCH_SIZE=128
train_dl = DataLoader(train_data,BATCH_SIZE,num_workers=4,pin_memory=True,shuffle=True)
test_dl = DataLoader(test_data,BATCH_SIZE,num_workers=4,pin_memory=True)

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def to_device(data,device):
    if isinstance(data,(list,tuple)):
        return [to_device(x,device) for x in data]
    return data.to(device,non_blocking=True)


class ToDeviceLoader:
    def __init__(self,data,device):
        self.data = data
        self.device = device
        
    def __iter__(self):
        for batch in self.data:
            yield to_device(batch,self.device)
            
    def __len__(self):
        return len(self.data)
    
device = get_device()
print(device)

train_dl = ToDeviceLoader(train_dl,device)
test_dl = ToDeviceLoader(test_dl,device)

def accuracy(predicted,actual):
    _, predictions = torch.max(predicted,dim=1)
    return torch.tensor(torch.sum(predictions==actual).item()/len(predictions))

class BaseModel(nn.Module):
    def training_step(self,batch):
        images,labels = batch
        out = self(images)
        loss = F.cross_entropy(out,labels)
        return loss
    
    def validation_step(self,batch,k_size=-1):
        images,labels = batch
        out = self(images,k_size=k_size)
        loss = F.cross_entropy(out,labels)
        acc = accuracy(out,labels)
        return {"val_loss":loss.detach(),"val_acc":acc}
    
    def validation_epoch_end(self,outputs):
        batch_losses = [loss["val_loss"] for loss in outputs]
        loss = torch.stack(batch_losses).mean()
        batch_accuracy = [accuracy["val_acc"] for accuracy in outputs]
        acc = torch.stack(batch_accuracy).mean()
        return {"val_loss":loss.item(),"val_acc":acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc']))
        
def conv_shortcut(in_channel,out_channel,stride):
    layers = [nn.Conv2d(in_channel,out_channel,kernel_size=(1,1),stride=(stride,stride)),
             nn.BatchNorm2d(out_channel)]
    return nn.Sequential(*layers)

def block(in_channel,out_channel,k_size,stride, conv=False):
    layers = None
    
    first_layers = [nn.Conv2d(in_channel,out_channel[0],kernel_size=(1,1),stride=(1,1)),
                    nn.BatchNorm2d(out_channel[0]),
                    nn.ReLU(inplace=True)]
    if conv:
        first_layers[0].stride=(stride,stride)
    
    second_layers = [nn.Conv2d(out_channel[0],out_channel[1],kernel_size=(k_size,k_size),stride=(1,1),padding=1),
                    nn.BatchNorm2d(out_channel[1])]

    layers = first_layers + second_layers
    
    return nn.Sequential(*layers)
    

class MResnet(BaseModel):
    
    def __init__(self,in_channels,num_classes):
        super().__init__()
        
        self.stg1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=64,kernel_size=(3),stride=(1),padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2)
        )
        
        ##stage 2
        self.convShortcut2 = conv_shortcut(64,256,1)
        
        self.conv2 = block(64,[64,256],3,1,conv=True)
        self.ident2 = block(256,[64,256],3,1)

        ##stage 3
        self.convShortcut3 = conv_shortcut(256,512,2)
        
        self.conv3 = block(256,[128,512],3,2,conv=True)
        self.ident3 = block(512,[128,512],3,2)

        ##stage 4
        self.convShortcut4 = conv_shortcut(512,1024,2)
        
        self.conv4 = block(512,[256,1024],3,2,conv=True)
        self.ident4 = block(1024,[256,1024],3,2)
    
        ##Classify
        self.classifier = nn.Sequential(
            nn.AvgPool2d(kernel_size=(4)),
            nn.Flatten(),
            nn.Linear(1024,num_classes)
        )
        
        self.avgpool = nn.AvgPool2d(kernel_size=(4))
        self.flatt = nn.Flatten()
        self.final = nn.Linear(1024, num_classes)
        
    def forward(self,inputs, k_size=-1):
        out = self.stg1(inputs)
        
        # stage 2
        out = F.relu(self.conv2(out) + self.convShortcut2(out))
        out = F.relu(self.ident2(out) + out)
        out = F.relu(self.ident2(out) + out)
        
        # stage3
        out = F.relu(self.conv3(out) + (self.convShortcut3(out)))
        out = F.relu(self.ident3(out) + out)
        out = F.relu(self.ident3(out) + out)
        out = F.relu(self.ident3(out) + out)
        
        # stage4             
        out = F.relu(self.conv4(out) + (self.convShortcut4(out)))
        out = F.relu(self.ident4(out) + out)
        out = F.relu(self.ident4(out) + out)
        out = F.relu(self.ident4(out) + out)
        out = F.relu(self.ident4(out) + out)
        out = F.relu(self.ident4(out) + out)

        out = self.avgpool(out)
        out = self.flatt(out)
        
        if k_size != -1:
            global kmeans
            activations = out
            for i in range(len(activations)):
                tensor = np.array(activations[i].tolist())
                closest_centroid = torch.tensor(find_closest_centroid(tensor, kmeans[k_size]), device='cuda') 
                out[i] = closest_centroid

        out = self.final(out)
        return out
        
model = MResnet(3,100)
model = to_device(model,device)
model.load_state_dict(torch.load("resnet128.pth"))

@torch.no_grad()
def evaluate(model,test_dl,k_size=-1):
    model.eval()
    outputs = [model.validation_step(batch,k_size=k_size) for batch in test_dl]
    return model.validation_epoch_end(outputs)

result = evaluate(model,test_dl)
print(result)
results = [result]
for cluster in clusters:
    result = evaluate(model,test_dl,cluster)
    results.append(results)
    print(f"{cluster:3d}: {result}")

