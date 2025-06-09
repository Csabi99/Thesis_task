import argparse
import flwr as fl
from logging import ERROR, INFO
import yaml
from flwr.common.logger import log
import datetime
from typing import Callable, Tuple, Dict, Optional
from client import CifarClassifier, test, set_parameters, load_cifar10
import torch
import strategies as flwr_strategies
import traceback


def init_config(path: Optional[str] = '/app/config.yaml') -> argparse.Namespace:
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description="Flower client")
    parser.add_argument(
        "--start_clients", type=int, default=config["start_clients"], help="Number of starting clients"
    )        
    parser.add_argument(
        "--min_clients", type=int, default=config["min_clients"], help="Number of clients"
    )    
    parser.add_argument(
        "--available_clients", type=int, default=config["available_clients"], help="Minimal number of clients to be available for federated learning to start"
    )        
    parser.add_argument(
        "--fraction_fit", type=float, default=config["fraction_fit"], help="Number of clients"
    )            
    parser.add_argument(
        "--number_of_rounds",
        type=int,
        default=config["rounds"],
        help="Number of FL rounds",
    )    
    parser.add_argument(
        '--d', action='store_true', default=config["docker"],
        help="Running in Docker-compose",
    )    
    parser.add_argument(
        '--strategy', type=str, default=config.get("strategy", {}).get("name", "FedAvg"),
        help="Federated learning strategy to use (FedAvg, FedProx, etc.). Gets from configfile. If not defined there, defaults to FedAvg.",
    )
    parser.add_argument(
        '--strategy_config', type=str, default=config.get("strategy", {}).get("params", {}),
        help="Federated learning strategy parameters to use. Gets from configfile. If not defined there, defaults to {}.",
    )
    args = parser.parse_args()
    return args


# Function to Start Federated Learning Server
def start_fl_server(strategy, rounds):
    try:
        fl.server.start_server(
            server_address="0.0.0.0:8080",
            config=fl.server.ServerConfig(num_rounds=rounds),
            strategy=strategy,
        )
    except Exception as e:
        log(ERROR, f"FL Server error: {traceback.format_exc()}")
        raise e

# could only use in  a custom strategy and then I could give the params here which I would store as class params initially
def fit_config(server_round: int):
    """Return training configuration dict for each round."""
    config = {
        "batch_size": 32,
        "current_round": server_round,
        "local_epochs": 2,
    }
    return config

# The `evaluate` function will be called by Flower after every round
def evaluate(
    server_round: int,
    parameters,
    config):
    net = CifarClassifier(torch.device("cpu")).to(torch.device("cpu"))
    trainloader, valloader = load_cifar10(32, 0, args.start_clients+1)
    set_parameters(net, parameters)  # Update model with the latest parameters
    loss, accuracy = test(net, valloader)
    print(f"Server-side evaluation loss {loss} / accuracy {accuracy}")
    return loss, {"accuracy": accuracy}

 



# Main Function
if __name__ == "__main__":
    args = init_config()
    start_time = datetime.datetime.now()
    fl.common.logger.configure(identifier=f"FlowerServer", filename=f"/app/logs/log_server_{args.strategy}.txt")
    log(INFO, f"{start_time}: Server script started with {args.start_clients} clients")
    # # Initialize Strategy Instance and Start FL Server
    # #strategy_instance = fl.server.strategy.FedAvg(min_fit_clients=args.min_clients, min_available_clients=args.available_clients, fraction_fit=args.fraction_fit, evaluate_fn=evaluate)
    strategy_class = getattr(flwr_strategies, args.strategy)
    strategy_instance = strategy_class(min_fit_clients=args.min_clients, min_available_clients=args.available_clients, fraction_fit=args.fraction_fit, evaluate_fn=evaluate, **args.strategy_config)
    log(INFO, f"Using strategy: {args.strategy}")
    log(INFO, f"Strategy parameters: {args.strategy_config}")
    start_fl_server(strategy=strategy_instance, rounds=args.number_of_rounds)
    end_time = datetime.datetime.now()
    duration = end_time-start_time
    log(INFO, f"{end_time}: Server script finished, took {duration.total_seconds()} seconds")