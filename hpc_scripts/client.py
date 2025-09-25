from collections import OrderedDict
from typing import List, Tuple
import argparse
from flwr.common.logger import log
from logging import INFO, DEBUG, ERROR
import yaml


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import os
from torchvision.io import read_image
from torch.utils.data import random_split
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import torchvision.models as models
import random
import flwr as fl
import datetime

class NuImageDataset(Dataset):
    def __init__(self, img_dir, is_test: bool = False):
        self.img_dir = img_dir
        self.transform = transforms.Compose([
            transforms.Resize(size=(80, 45)),  # Resize images to 80x45 pixels
            transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip images horizontally with a probability of 50%
            transforms.RandomRotation(degrees=15),  # Randomly rotate images within ±15 degrees
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Randomly adjust color properties
            #transforms.ToTensor(),  # Convert PIL images to PyTorch tensors
            transforms.ConvertImageDtype(torch.float32),  # <--- convert uint8 → float32 in [0,1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])  # Normalize
        ])
        self.test_transform = transforms.Compose([
            transforms.Resize(size=(80, 45)),  # Resize images to 80x45 pixels
            #transforms.ToTensor(),  # Convert PIL images to PyTorch tensors
            transforms.ConvertImageDtype(torch.float32),  # <--- convert uint8 → float32 in [0,1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])  # Normalize
        ])
        self.test = is_test
        self.images = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
        self.images.sort()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        #img_path = os.path.join(self.img_dir, f'{self.img_labels.iloc[idx, 0]}.png')
        img_path = os.path.join(self.img_dir, self.images[idx])
        image = read_image(img_path)
        # label = self.img_labels.iloc[idx, 1]
        label  = random.randint(0, 9)  # Dummy label for illustration
        if self.test:
            image = self.test_transform(image)
        else:
            image = self.transform(image)
        return image, label  
    
def load_nu(batch_size, i, n):
    dataset = NuImageDataset(img_dir='/app/data', is_test=False)
    test_dataset = NuImageDataset(img_dir='/app/data', is_test=True)
    log(INFO, f"NuDataset")
    # generator1 = torch.Generator().manual_seed(42)
    # fractions = [1 / n for _ in range(n - 1)]  # First n-1 elements
    # fractions.append(1 - sum(fractions))       # Adjust the last element
    # tr_splits = random_split(dataset, fractions, generator=generator1)
    # te_splits = random_split(test_dataset, fractions, generator=generator1)
    # train_dataloader = DataLoader(tr_splits[i], batch_size=batch_size, shuffle=False)
    # test_dataloader = DataLoader(te_splits[i], batch_size=batch_size, shuffle=False)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)    
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
    train_dataloader = DataLoader(tr_splits[i], batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(te_splits[i], batch_size=batch_size, shuffle=False)
    return train_dataloader, test_dataloader
    
class CifarClassifier(nn.Module):
    def __init__(self, dev):
        super(CifarClassifier, self).__init__()
        
        # Two convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(32 * 2 * 2, 64)  # Adjusted for downsampled dimensions
        self.fc2 = nn.Linear(64, 10)  # Output layer for binary classification

        # Dropout layer for regularization
        self.dropout = nn.Dropout(0.5)
        self._dev = dev
        self._global_model = None
        self._run_mode = None

    def forward(self, x):
        # Convolutional layers with ReLU and max-pooling
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)  # Flatten tensor

        # Fully connected layers with dropout and ReLU activation
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        # Output with sigmoid activation for binary classification
        return x    

    def set_global_model(self, model):
        """Set the global model for the client."""
        self._global_model = model
    

class NuClassifier(nn.Module):
    def __init__(self, dev, input_shape=(3, 80, 45)):
        super(NuClassifier, self).__init__()
        log(INFO, f"NuClasssifier")
        # Convs
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        
        # figure out flatten size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)  # batch=1, shape=(3,80,45)
            dummy_out = self._forward_convs(dummy)
            flatten_dim = dummy_out.view(1, -1).size(1)
        
        # FC layers
        self.fc1 = nn.Linear(flatten_dim, 64)
        self.fc2 = nn.Linear(64, 10)

        self.dropout = nn.Dropout(0.5)
        self._dev = dev
        self._global_model = None
        self._run_mode = None

    def _forward_convs(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        return x

    def forward(self, x):
        x = self._forward_convs(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def set_global_model(self, model):
        """Set the global model for the client."""
        self._global_model = model


class TransferClassifier(nn.Module):
    def __init__(self, dev, num_classes=10, freeze_backbone=True):
        super(TransferClassifier, self).__init__()
        
        # Load MobileNetV2 backbone
        self.backbone = models.mobilenet_v2(pretrained=True)

        # Optionally freeze backbone
        if freeze_backbone:
            for param in self.backbone.features.parameters():
                param.requires_grad = False

        # Replace classifier head
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

        self._dev = dev
        self._global_model = None
        self._run_mode = None

    def forward(self, x):
        return self.backbone(x)

    def set_global_model(self, model):
        self._global_model = model


def train(net, trainloader, epochs: int, verbose=False, config=None):
    """Train the network on the training set."""
    # if net._run_mode == "fedcm":
    #     #https://pytorch.org/docs/stable/generated/torch.optim.SGD.html - based on this
    #     optimizer = torch.optim.SGD(net.parameters(), lr=config["eta_l"])
    # else:
    #     optimizer = torch.optim.Adam(net.parameters())
    optimizer = torch.optim.Adam(net.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for batch in trainloader:
            images, labels = batch[0].to(net._dev), batch[1].to(net._dev)
            optimizer.zero_grad()
            outputs = net(images)        
            if net._run_mode == "fedprox":
                proximal_term = 0.0
                for local_weights, global_weights in zip(net.parameters(), net._global_model):
                    np.save("/app/logs/local_weights.npy", local_weights.detach().numpy())
                    np.save("/app/logs/global_weights.npy", global_weights) 
                    current = np.linalg.norm((local_weights.detach().numpy() - global_weights))**2
                    proximal_term += current
                loss = criterion(net(images), labels) + (config["proximal_mu"] / 2) * proximal_term
            else:
                loss = criterion(outputs, labels)
            loss.backward()
            if net._run_mode == "fedcm":
                with torch.no_grad():
                    g = [param.grad for param in net.parameters() if param.grad is not None]
                    if config["gradient"] == None:
                        new_g = g
                    else:
                        new_g = [config["decay_alfa"] * _g + (1-config["decay_alfa"]) * dt for dt, _g in zip(config["gradient"], g)]
                    for param, custom_grad in zip(net.parameters(), new_g):
                        param.grad = custom_grad.clone().to(param.device)                      

            optimizer.step()
            # Metrics
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        if verbose:
            log(INFO, f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")


# def train(net, trainloader, epochs: int, verbose=False):
#     """Train the network on the training set."""
#     criterion = torch.nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(net.parameters())
#     net.train()
#     for epoch in range(epochs):
#         correct, total, epoch_loss = 0, 0, 0.0
#         for batch in trainloader:
#             images, labels = batch[0].to(net._dev), batch[1].to(net._dev)
#             optimizer.zero_grad()
#             outputs = net(images)        
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             # Metrics
#             epoch_loss += loss
#             total += labels.size(0)
#             correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
#         epoch_loss /= len(trainloader.dataset)
#         epoch_acc = correct / total
#         if verbose:
#             log(INFO, f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")

def test(net, testloader):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            images, labels = batch[0].to(net._dev), batch[1].to(net._dev)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy

def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def get_head_parameters(model):
    return [val.cpu().numpy() for _, val in model.backbone.classifier.state_dict().items()]

def set_head_parameters(model, parameters):
    state_dict = model.backbone.classifier.state_dict()
    new_state_dict = {k: torch.tensor(v) for k, v in zip(state_dict.keys(), parameters)}
    model.backbone.classifier.load_state_dict(new_state_dict)

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, net, trainloader, valloader, epochs, learning_rate):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.epochs = epochs
        self.learning_rate = learning_rate

    def get_parameters(self, config):
        return get_head_parameters(self.net)
        # return get_parameters(self.net)

    def fit(self, parameters, config):
        #self.net.set_global_model(parameters)
        if 'proximal_mu' in config.keys():
            log(INFO, f"Using run mode: fedprox")
            self.net.set_global_model(parameters)
            self.net._run_mode = "fedprox"
        elif config.get("mode", None) == "FedCM":
            log(INFO, f"Using run mode: fedcm")
            grad_length = config["grad_length"]
            if not config["grad"]:
                config["gradient"] = None
            else:
                config["gradient"] = parameters[:grad_length*(-1)]
            parameters = parameters[:grad_length]
            self.net.set_global_model(parameters)
            self.net._run_mode = "fedcm"
        else:
            log(INFO, f"Using run mode: default")
            self.net._run_mode = "default"
        # set_parameters(self.net, parameters)
        set_head_parameters(self.net, parameters)
        #pairs = [{"weight1": w1.detach().numpy(), "weight2": w2} for w1, w2 in zip(net.parameters(), parameters)]
        #log(DEBUG, f"possible input: {pairs}")
        if self.learning_rate != 0:
            config["eta_l"] = self.learning_rate
        #train(self.net, self.trainloader, epochs=self.epochs, verbose=True, config=config)
        train(self.net, self.trainloader, epochs=self.epochs, verbose=True, config=config)
        #return get_parameters(self.net), len(self.trainloader), {}
        return get_head_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        # set_parameters(self.net, parameters)
        set_head_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader)
        #self.logger.info("accuracy: "+ str(float(accuracy)))
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}    

class DummyClient(fl.client.NumPyClient):
    def __init__(self, net, trainloader, valloader, epochs, lr):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.epochs = epochs
        self.lr = lr

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):

        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader)
        #self.logger.info("accuracy: "+ str(float(accuracy)))
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}  

def init_client():
    with open('/app/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description="Flower client")
    parser.add_argument(
        "--server_address", type=str, default=config["server"], help="Address of the server"
    )
    parser.add_argument(
        "--start_clients", type=int, default=config["start_clients"], help="Number of starting clients"
    )        
    parser.add_argument(
        "--batch_size", type=int, default=config["batch_size"], help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=config["learning_rate"], help="Learning rate for the optimizer"
    )
    parser.add_argument(
        "--epochs", type=float, default=config["epochs"], help="Number of epochs while training"
    )
    parser.add_argument(
        '--d', action='store_true', default=config["docker"],
        help="Running in Docker-compose",
    )    
    parser.add_argument(
        '--dir', type=str, default=config["working_dir"],
        help="If not running in Docker-compose, you can specify the directory where pictures and their labels can be found",
    ) 
    parser.add_argument("--noise", help=f"the client is like a random baseline model",
                        action='store_true', default=False)
    parser.add_argument("--num", help=f"the client starts training based on test case number: from 1",
                        type=int)

    args = parser.parse_args()
    return args
    
if __name__ == "__main__":
    args = init_client()
    NUM_CLIENTS = args.start_clients
    BATCH_SIZE = args.batch_size
    if args.num > NUM_CLIENTS:
        log(ERROR, "Test case number does not exist")
        exit()
    fl.common.logger.configure(identifier=f"FlowerClient_{args.num}", filename=f"/app/logs/log_Client_{args.num}.txt")
    start_time = datetime.datetime.now()
    log(INFO, f"{start_time}: Client script started")
    DEVICE = torch.device("cpu")
    # trainloader, valloader = load_cifar10(args.batch_size, args.num, args.start_clients+1)
    # net = CifarClassifier(DEVICE).to(DEVICE)
    trainloader, valloader = load_nu(args.batch_size, args.num, args.start_clients+1)
    net = TransferClassifier(DEVICE).to(DEVICE)
    if args.noise:
        flwr_client = DummyClient(net, trainloader, valloader, args.epochs, args.learning_rate).to_client()        
    else:
        flwr_client = FlowerClient(net, trainloader, valloader, args.epochs, args.learning_rate).to_client()
    fl.client.start_client(server_address=args.server_address, client=flwr_client.to_client())
    end_time = datetime.datetime.now()
    duration = end_time-start_time
    log(INFO, f"{datetime.datetime.now()}: Client script finished, took {duration.total_seconds()} seconds")