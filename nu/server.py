import argparse
import flwr as fl
from logging import ERROR, INFO
import yaml
from flwr.common.logger import log
import datetime
from typing import Callable, Tuple, Dict, Optional
from client import test
import torch
import strategies as flwr_strategies
import traceback
import wandb
import openml
import json
import os
from models import TransferClassifier, LightweightClassifier
import load_data as flwr_data

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
    parser.add_argument(
        '--classifier', type=str, default=config.get("classifier", "TransferClassifier"), # TODO update config files accordingly (TransferClassifier or LightweightClassifier)
        help="Classifier to use. Gets from configfile. If not defined there, defaults to TransferClassifier (MobileNetV2), which is loaded from /app/models.",
    )    
    args = parser.parse_args()
    return args


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
    return

# # here you could add a piece of code that would patch any strategy class to save the model itself to a file
# # # Save PyTorch model to file
# # model_path = "global_model.pt"
# # torch.save(model.state_dict(), model_path)
# def patch_strategy(cls):
#     """Patch a strategy class so its evaluate logs to WandB."""
#     old_evaluate = getattr(cls, "evaluate", None)

#     def custom_evaluate(self, server_round, parameters):
#         # call original if it exists
#         eval_res = None
#         if old_evaluate is not None:
#             eval_res = old_evaluate(self, server_round, parameters)

#             # handle None metrics safely
#             loss, metrics = eval_res
#             metrics = metrics or {}
#         else:
#             loss, metrics = None, {}

#         # log to wandb
#         wandb.log({
#             "round": server_round,
#             "loss": loss,
#             **metrics,
#         })

#         return eval_res

#     cls.evaluate = custom_evaluate
#     return cls

# The `evaluate` function will be called by Flower after every round
# TODO: inject the classifier and its evaluation from a different file which is model specific. hard because params are quite strict here: https://github.com/adap/flower/blob/main/framework/py/flwr/server/strategy/fedavg.py line 167
def transfer_evaluate(
    server_round: int,
    parameters,
    config):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = TransferClassifier(DEVICE, checkpoint=True)
    trainloader, valloader = flwr_data.load_nu(32, 0, args.start_clients+1)
    net.set_head_parameters(parameters)  # Update model with the latest parameters
    model_path = "/app/logs/global_transfer_model.pt"
    torch.save(net.state_dict(), model_path)
    loss, accuracy = test(net, valloader)
    print(f"Server-side evaluation loss {loss} / accuracy {accuracy}")
    wandb.log({
        "round": server_round,
        "loss": loss,
        "accuracy": accuracy
    })
    return loss, {"accuracy": accuracy}

def lightweight_evaluate(
    server_round: int,
    parameters,
    config):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = LightweightClassifier(DEVICE)
    trainloader, valloader = flwr_data.load_nu(32, 0, args.start_clients+1)
    net.set_head_parameters(parameters)  # Update model with the latest parameters
    model_path = "/app/logs/global_lightweight_model.pt"
    torch.save(net.state_dict(), model_path)
    loss, accuracy = test(net, valloader)
    print(f"Server-side evaluation loss {loss} / accuracy {accuracy}")
    wandb.log({
        "round": server_round,
        "loss": loss,
        "accuracy": accuracy
    })
    return loss, {"accuracy": accuracy}
 



# Main Function
if __name__ == "__main__":
    args = init_config()
    start_time = datetime.datetime.now()
    fl.common.logger.configure(identifier=f"FlowerServer", filename=f"/app/logs/log_server_{args.strategy}.txt")
    log(INFO, f"{start_time}: Server script started with {args.start_clients} clients")
    wandb.init(project="federated-learning", group="experiment-2", job_type="server", config={
        "clients_num": args.start_clients,
        "rounds": args.number_of_rounds,
        "strategy": args.strategy,
    })    
    strategy_class = getattr(flwr_strategies, args.strategy)
    if args.classifier == "TransferClassifier":
        evaluate_fn = transfer_evaluate
    elif args.classifier == "LightweightClassifier":
        evaluate_fn = lightweight_evaluate
    else:
        raise ValueError(f"Unknown classifier: {args.classifier}")
    # strategy_class = patch_strategy(strategy_class) # TODO - do we need patching? probably not for evaluation, but for saving model, maybe yes?
    strategy_instance = strategy_class(min_fit_clients=args.min_clients, min_available_clients=args.available_clients, fraction_fit=args.fraction_fit, evaluate_fn=evaluate_fn, **args.strategy_config)
    log(INFO, f"Using strategy: {args.strategy}")
    log(INFO, f"Strategy parameters: {args.strategy_config}")
    start_fl_server(strategy=strategy_instance, rounds=args.number_of_rounds)
    end_time = datetime.datetime.now()
    duration = end_time-start_time
    log(INFO, f"{end_time}: Server script finished, took {duration.total_seconds()} seconds")