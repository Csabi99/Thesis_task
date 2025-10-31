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
import flwr as fl
import datetime
import wandb
import models as flwr_models    
import load_data as flwr_data

def train(net, trainloader, epochs: int, verbose=False, config=None, valloader=None):
    """Train the network on the training set with optional LR scheduling."""
    optimizer = torch.optim.Adam(net.parameters())
    # optimizer = torch.optim.Adam(net.parameters(), lr=config.get("learning_rate", 1e-3))
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer,
    #     mode='min',
    #     patience=3,
    #     factor=0.5,
    #     min_lr=1e-6  # prevents learning rate from vanishing
    # )
    criterion = torch.nn.CrossEntropyLoss()
    net.train()

    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0

        for images, labels in trainloader:
            images, labels = images.to(net._dev), labels.to(net._dev)
            optimizer.zero_grad()
            outputs = net(images)

            # Handle FedProx loss
            if net._run_mode == "fedprox":
                proximal_term = 0.0
                for local_weights, global_weights in zip(net.parameters(), net._global_model):
                    local_np = local_weights.detach().cpu().numpy()
                    global_np = global_weights if isinstance(global_weights, np.ndarray) else np.array(global_weights)
                    proximal_term += np.linalg.norm(local_np - global_np) ** 2
                loss = criterion(outputs, labels) + (config["proximal_mu"] / 2) * proximal_term
            else:
                loss = criterion(outputs, labels)

            loss.backward()

            if net._run_mode == "fedcm":
                with torch.no_grad():
                    grads = [p.grad for p in net.parameters() if p.grad is not None]
                    if config.get("gradient") is None:
                        new_grads = grads
                    else:
                        new_grads = [
                            config["decay_alfa"] * g + (1 - config["decay_alfa"]) * old
                            for old, g in zip(config["gradient"], grads)
                        ]
                    for param, g in zip(net.parameters(), new_grads):
                        param.grad = g.clone().to(param.device)

            optimizer.step()

            # Metrics
            epoch_loss += loss.item()
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()

        # --- End of epoch ---
        epoch_loss /= len(trainloader)
        epoch_acc = correct / total if total > 0 else 0.0
        val_loss = 0.0
        accuracy = 0.0
        # Optional validation loss for scheduler
        if valloader is not None:
            net.eval()
            
            with torch.no_grad():
                for images, labels in valloader:
                    images, labels = images.to(net._dev), labels.to(net._dev)
                    outputs = net(images)
                    val_loss += criterion(outputs, labels).item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()                    
            val_loss /= len(valloader)
            #scheduler.step(val_loss)
            net.train()
            accuracy = correct / total if total > 0 else 0.0            
        #else:
            #scheduler.step(epoch_loss)

        log(INFO, f"Epoch {epoch+1}: train loss {epoch_loss:.4f}, acc {epoch_acc:.3f}, lr {optimizer.param_groups[0]['lr']:.6f}")
        if valloader is not None:
            log(INFO, f"           val loss {val_loss:.4f}, acc {accuracy:.3f}")

        if config.get("verbose", False):
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": epoch_loss,
                "train_acc": epoch_acc,
                "lr": optimizer.param_groups[0]['lr']
            })



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
    # normalize by number of samples
    loss /= len(testloader.dataset)
    accuracy = correct / total if total > 0 else 0.0
    log(INFO, f"Validation: val loss {loss}, accuracy {accuracy}")
    return loss, accuracy

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, net, trainloader, valloader, epochs, learning_rate, verbose=True):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.verbose = verbose

    def get_parameters(self, config):
        return self.net.get_parameters()

    def fit(self, parameters, config):
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
        self.net.set_parameters(parameters)
        if self.learning_rate != 0:
            config["eta_l"] = self.learning_rate
        config = {**config, "learning_rate": self.learning_rate, "verbose": self.verbose}
        train(self.net, self.trainloader, epochs=self.epochs, verbose=True, config=config, valloader=self.valloader)
        return self.net.get_parameters(), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        self.net.set_parameters(parameters)
        loss, accuracy = test(self.net, self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}    

class DummyClient(fl.client.NumPyClient):
    def __init__(self, net, trainloader, valloader, epochs, lr):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.epochs = epochs
        self.lr = lr

    def get_parameters(self, config):
        return self.net.get_parameters()

    def fit(self, parameters, config):
        return self.net.get_parameters(), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        self.net.set_parameters(parameters)
        loss, accuracy = test(self.net, self.valloader)
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
    parser.add_argument(
        '--strategy', type=str, default=config.get("strategy", {}).get("name", "FedAvg"),
        help="Federated learning strategy to use (FedAvg, FedProx, etc.). Gets from configfile. If not defined there, defaults to FedAvg.",
    )    
    parser.add_argument(
        '--classifier', type=str, default=config.get("classifier", "TransferClassifier"), # TODO update config files accordingly (TransferClassifier or LightweightClassifier)
        help="Classifier to use. Gets from configfile. If not defined there, defaults to TransferClassifier (MobileNetV2), which is loaded from /app/models.",
    )
    parser.add_argument(
        '--input_image_width', type=int, default=config.get("input_image_width", 80),
        help="Input image width. Gets from configfile. If not defined there, defaults to 80.",
    )
    parser.add_argument(
        '--input_image_height', type=int, default=config.get("input_image_height", 45),
        help="Input image height. Gets from configfile. If not defined there, defaults to 45.",
    )
    parser.add_argument(
        '--num_classes', type=int, default=config.get("num_classes", 4),
        help="Number of output classes. Gets from configfile. If not defined there, defaults to 4.",
    )
    parser.add_argument(
        '--not_model_checkpoint', help="Don't use model checkpoint. Start training from scratch",
        action='store_true'
    )
    args = parser.parse_args()
    return args
    
if __name__ == "__main__":
    args = init_client()
    NUM_CLIENTS = args.start_clients
    BATCH_SIZE = args.batch_size
    wandb.init(
    project=f"federated-learning-{args.strategy}-{args.start_clients}", # project name
    group="experiment-2",          # same as server
    job_type="client",             # marks this as a client run
    name=f"client-{args.num}"
    )

    if args.num > NUM_CLIENTS:
        log(ERROR, "Test case number does not exist")
        exit()
    fl.common.logger.configure(identifier=f"FlowerClient_{args.num}", filename=f"/app/logs/log_Client_{args.num}.txt")
    start_time = datetime.datetime.now()
    log(INFO, f"{start_time}: Client script started")

    # Device selection: prefer GPU if available
    if torch.cuda.is_available():
        log(INFO, "CUDA is available. Using GPU.")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainloader, valloader = flwr_data.load_nu(args.batch_size, args.num, args.start_clients+1, width=args.input_image_width, height=args.input_image_height)
    net_class = getattr(flwr_models, args.classifier)
    #net = net_class(DEVICE, checkpoint=not args.not_model_checkpoint, input_shape=(3, args.input_image_height, args.input_image_width), num_classes=args.num_classes)
    net = net_class(DEVICE, checkpoint=args.not_model_checkpoint, input_shape=(3, args.input_image_height, args.input_image_width), num_classes=args.num_classes)

    if args.noise:
        flwr_client = DummyClient(net, trainloader, valloader, args.epochs, args.learning_rate).to_client()        
    else:
        flwr_client = FlowerClient(net, trainloader, valloader, args.epochs, args.learning_rate).to_client()
    fl.client.start_client(server_address=args.server_address, client=flwr_client.to_client())
    end_time = datetime.datetime.now()
    duration = end_time-start_time
    log(INFO, f"{datetime.datetime.now()}: Client script finished, took {duration.total_seconds()} seconds")