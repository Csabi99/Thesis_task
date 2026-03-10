from models import TransferClassifier, LightweightClassifier, TinyClassifier
from load_data import load_nu, load_cifar10 as load_cifar
from client import FlowerClient
import torch
from flwr.common.logger import log
import flwr as fl
from logging import INFO
import datetime



if __name__ == "__main__":
    #fl.common.logger.configure(identifier=f"Base_Model", filename=f"/app/logs/Base_Model.txt")
    start_time = datetime.datetime.now()
    log(INFO, f"{start_time}: Script started")
    num = 1
    epochs = 300
    batch_size = 32
    if torch.cuda.is_available():
        log(INFO, "CUDA is available. Using GPU.")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainloader, valloader = load_nu(batch_size, num, 32, federated=False)
    net = TinyClassifier(DEVICE, input_shape=(3, 80, 45), num_classes=4)
    #net = LightweightClassifier(DEVICE, input_shape=(3, 120, 120), output_dim=4)
    client = FlowerClient(net, trainloader, valloader, epochs, 0.001, False)
    losses = []
    accuracies = []
    for i in range(1):
        client.fit(client.get_parameters({}), {})
        loss, val_len, accuracy = client.evaluate(client.get_parameters({}), {})
        losses.append(loss)
        accuracies.append(accuracy)
    end_time = datetime.datetime.now()
    duration = end_time-start_time
    log(INFO, f"{datetime.datetime.now()}: Client script finished, took {duration.total_seconds()} seconds")    
    log(INFO, f"Accuracy: {accuracies}, loss: {losses}")