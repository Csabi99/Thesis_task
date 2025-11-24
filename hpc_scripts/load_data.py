import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split
from torchvision.io import read_image
import pandas as pd
import os
from torchvision.datasets import CIFAR10
import torch
from flwr.common.logger import log
from logging import INFO
from PIL import Image
import random


class NuImageDataset(Dataset):
    def __init__(self, img_dir, cache_dir = None, width = 80, height = 45, is_test: bool = False):
        self.img_dir = img_dir
        self.cached = False
        self.test = is_test
        self.img_labels = pd.read_csv(os.path.join(img_dir, 'labels.csv'))    
        self.label_map = {
            "human": 0,
            "human-vehicle": 1,
            "none": 2,
            "vehicle": 3
        }            
        if cache_dir is not None and os.path.exists(cache_dir):
            self.cached = True
            self.cache_dir = cache_dir
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                    saturation=0.2, hue=0.1),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])                                           
            ])
            self.test_transform = transforms.Compose([
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])                 
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),  
                transforms.ConvertImageDtype(torch.float32),  # convert first -> values in [0,1]
                transforms.Resize(size=(width, height)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])                   
            ])
            self.test_transform = transforms.Compose([
                transforms.ToTensor(),  
                transforms.ConvertImageDtype(torch.float32),
                #transforms.Resize(size=(width, height)),  # Resize images
                #transforms.ToTensor(),  # Convert PIL images to PyTorch tensors
                #transforms.Lambda(lambda x: x / 255.0),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])                              
            ])        

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        if self.cached:
            image = torch.load(os.path.join(self.cache_dir, f"{self.img_labels.iloc[idx, 1].split('.')[0]}.pt"))
        else:
            img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 1])
            image = Image.open(img_path).convert("RGB")
        if self.test:
            image = self.test_transform(image)
        else:
            image = self.transform(image)
        str_label = self.img_labels.iloc[idx, 2]
        label = self.label_map[str_label]        
        return image, label  
    
def load_nu(batch_size, i, n, federated=True, width=80, height=45):
    train_dataset = NuImageDataset(img_dir='/app/data', cache_dir='/app/cache', width=width, height=height, is_test=False)
    test_dataset = NuImageDataset(img_dir='/app/data', cache_dir='/app/cache', width=width, height=height, is_test=True)
    log(INFO, f"NuDataset")
    generator1 = torch.Generator().manual_seed(42)
    train_splits = random_split(train_dataset, [0.8, 0.2], generator=generator1)    
    test_splits = random_split(test_dataset, [0.8, 0.2], generator=generator1)    
    if not federated:
        full_dataset = NuImageDataset('/app/data', cache_dir='/app/cache', width=width, height=height,is_test=False)
        generator = torch.Generator().manual_seed(42)
        train_idx, test_idx = random_split(range(len(full_dataset)), [0.8, 0.2], generator=generator)

        train_dataset = torch.utils.data.Subset(NuImageDataset('/app/data', cache_dir='/app/cache', width=width, height=height, is_test=False), train_idx)
        test_dataset  = torch.utils.data.Subset(NuImageDataset('/app/data', cache_dir='/app/cache', width=width, height=height, is_test=True),  test_idx)
        return DataLoader(train_dataset, batch_size=batch_size, shuffle=True), DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    if n > 10:
        n = 10
        i = random.randint(0, n-1)
    fractions = [1 / n for _ in range(n - 1)]  # First n-1 elements
    fractions.append(1 - sum(fractions))       # Adjust the last element
    tr_splits = random_split(train_splits[0], fractions)
    te_splits = random_split(test_splits[1], fractions)
    train_dataloader = DataLoader(tr_splits[i], batch_size=batch_size, num_workers=0)
    test_dataloader = DataLoader(te_splits[i], batch_size=batch_size, num_workers=0)
    return train_dataloader, test_dataloader

def load_cifar10(batch_size, i, n):
    transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip images horizontally with a probability of 50%
    transforms.RandomRotation(degrees=15),  # Randomly rotate images within ±15 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Randomly adjust color properties
    transforms.ToTensor(),  # Convert PIL images to PyTorch tensors
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ])
    test_transform = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL images to PyTorch tensors
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ])
    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    generator1 = torch.Generator().manual_seed(42)
    fractions = [1 / n for _ in range(n - 1)]  # First n-1 elements
    fractions.append(1 - sum(fractions))       # Adjust the last element
    tr_splits = random_split(train_dataset, fractions, generator=generator1)
    te_splits = random_split(test_dataset, fractions, generator=generator1)
    train_dataloader = DataLoader(tr_splits[i], batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(te_splits[i], batch_size=batch_size, shuffle=True)
    return train_dataloader, test_dataloader