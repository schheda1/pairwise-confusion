# import cv2
import numpy as np
import matplotlib.pyplot as plt
# import glob

import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

# import gc
import pandas as pd
# from PIL import Image

from torchvision import models
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset, DataLoader

# from torchsummary import summary

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')


class CubsDataset(Dataset):
    base_dir = 'CUB_200_2011/images'

    def __init__(self, root, train=True, transform=None, loader=default_loader):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.train = train
        self.loader = default_loader
        self.__load_metadata__()
        print('init')

    def __load_metadata__(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['image_id', 'image_name'])
        labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'), sep=' ',
                             names=['image_id', 'class_id'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'), sep=' ',
                                       names=['image_id', 'is_training_image'])
        class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'classes.txt'), sep=' ',
                                   names=['class_id', 'class_name'])

        data = images.merge(labels, on='image_id')
        self.data = data.merge(train_test_split, on='image_id')

        if self.train:
            self.data = self.data[self.data.is_training_image == 1]
        else:
            self.data = self.data[self.data.is_training_image == 0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data.iloc[index]
        path = os.path.join(self.root, self.base_dir, sample.image_name)
        image = self.loader(path)
        # image = image/255
        class_id = sample.class_id - 1

        # print(self.transform)
        if self.transform:
            image = self.transform(image)

        return image, class_id


transforms_example = transforms.Compose([transforms.Resize((224, 224)),
                                         transforms.ToTensor()
                                         ])

cubs_dataset_train = CubsDataset(root='datasets', transform=transforms_example)  # train #can use loader=pil_image
cubs_dataset_test = CubsDataset(root='datasets', train=False, transform=transforms_example)
print(len(cubs_dataset_train))
print(len(cubs_dataset_test))
print('loaded train and test sets')

### create validation set

batch_size = 30
validation_split = .05
shuffle_dataset = True
random_seed= 43
dataset_size = len(cubs_dataset_train)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)



checkpoint = torch.load('model/js_new_run/modeljs_12.pth')
# resnet = models.resnet50(pretrained=False)
# resnset.fc = nn.Sequential(
#        nn.Linear(2048, 200, bias=True)
#        )
resnet = checkpoint['model']
resnet.load_state_dict(checkpoint['state_dict'])



resnet.eval()  # eval mode 
with torch.no_grad():
    correct = 0
    total = 0
    D3 = DataLoader(cubs_dataset_test, batch_size=60, shuffle=False)
    for ind, (images, labels) in enumerate(D3):
        images = images.to(device)
        labels = labels.to(device)
        outputs = resnet(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        print(ind)

    print('Test Accuracy of the model: {} %'.format(100*correct / total))

# ls model -al
'''
checkpoint = {
    'model': resnet, 
    'state_dict': resnet.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scheduler': scheduler.state_dict()
}

torch.save(checkpoint, 'model/resnet_eval_model_cub.pth')

torch.save(resnet, 'model/resnet_cub.pth')
'''
