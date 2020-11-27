import torch
from torchvision import datasets, transforms
from torch.utils import data
import torchvision
import os
import matplotlib.pyplot as plt

data_dir = "DataSets"
data_transform = {x: transforms.Compose([transforms.Scale([224, 224]),
                                         transforms.ToTensor()])
                  for x in ["train", "test"]}  # 将图片统一处理为224*224大小并变为tensor
image_dataSets = {x: datasets.ImageFolder(root=os.path.join(data_dir, x),
                                          transform=data_transform[x])
                  for x in ["train", "test"]}
data_loader = {x: torch.utils.data.DataLoader(dataset=image_dataSets[x],
                                              batch_size=16,
                                              shuffle=True)
               for x in ["train", "test"]}

# 返回处理之后的数据集
def dataLoader():
    return data_loader
    