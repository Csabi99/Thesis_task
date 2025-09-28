import argparse
import flwr as fl
from logging import ERROR, INFO
import yaml
from flwr.common.logger import log
import datetime
from typing import Callable, Tuple, Dict, Optional
from client import CifarClassifier, NuClassifier, TransferClassifier, test, set_parameters, set_head_parameters, load_cifar10, load_nu
import torch
import strategies as flwr_strategies
import traceback
import wandb
import openml
import json
import os

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
        hist = fl.server.start_server(
            server_address="0.0.0.0:8080",
            config=fl.server.ServerConfig(num_rounds=rounds),
            strategy=strategy,
        )
        # run.evaluations = {
        #     "final_accuracy": hist.metrics_centralized["accuracy"][-1][1] if "accuracy" in hist.metrics_centralized else None,
        #     "final_loss": hist.losses_centralized[-1][1],
        #     "rounds": rounds,
        #     "min_fit_clients": strategy.min_fit_clients,
        #     "min_available_clients": strategy.min_available_clients,
        #     "fraction_fit": strategy.fraction_fit,
        #     "strategy": strategy.__class__.__name__,
        #     "loss": json.dumps(hist.losses_centralized),
        #     "metrics": json.dumps(hist.metrics_centralized),
        # }

        # run.publish()
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

# here you could add a piece of code that would patch any strategy class to save the model itself to a file
# # Save PyTorch model to file
# model_path = "global_model.pt"
# torch.save(model.state_dict(), model_path)

# # Create a run
# run = openml.runs.OpenMLRun()
# run.evaluations = {
#     "final_accuracy": history[-1]["acc"],
#     "final_loss": history[-1]["loss"],
# }

# # Upload model file as an artifact
# run.attach_file(model_path)

# # Publish run
# run.publish()
def patch_strategy(cls):
    """Patch a strategy class so its evaluate logs to WandB."""
    old_evaluate = getattr(cls, "evaluate", None)

    def custom_evaluate(self, server_round, parameters):
        # call original if it exists
        eval_res = None
        if old_evaluate is not None:
            eval_res = old_evaluate(self, server_round, parameters)

            # handle None metrics safely
            loss, metrics = eval_res
            metrics = metrics or {}
        else:
            loss, metrics = None, {}

        # log to wandb
        wandb.log({
            "round": server_round,
            "loss": loss,
            **metrics,
        })

        return eval_res

    cls.evaluate = custom_evaluate
    return cls

# The `evaluate` function will be called by Flower after every round
def evaluate(
    server_round: int,
    parameters,
    config):
    # net = CifarClassifier(torch.device("cpu")).to(torch.device("cpu"))
    net = TransferClassifier(torch.device("cpu")).to(torch.device("cpu"))
    # trainloader, valloader = load_cifar10(32, 0, args.start_clients+1)
    trainloader, valloader = load_nu(32, 0, args.start_clients+1)
    # net = NuClassifier(torch.device("cpu")).to(torch.device("cpu"))
    # trainloader, valloader = load_nu(32, 0, args.start_clients+1)    
    # set_parameters(net, parameters)  # Update model with the latest parameters
    set_head_parameters(net, parameters)  # Update model with the latest parameters
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
    # run = openml.runs.OpenMLRun(
    #     task_id=None,   # optional if not tied to an OpenML task
    #     flow_id=8053,   # optional if not tied to a model
    #     dataset_id=None,  # ID of the Fashion-MNIST dataset on OpenML
    #     tags=["federated-learning", "flower", args.strategy, "test"],
    #     parameter_settings={
    #         "rounds": args.number_of_rounds,
    #         "min_fit_clients": args.min_clients,
    #         "min_available_clients": args.available_clients,
    #         "fraction_fit": args.fraction_fit,
    #         "strategy": args.strategy,
    #     }
    # )    
    # # Initialize Strategy Instance and Start FL Server
    # #strategy_instance = fl.server.strategy.FedAvg(min_fit_clients=args.min_clients, min_available_clients=args.available_clients, fraction_fit=args.fraction_fit, evaluate_fn=evaluate)
    strategy_class = getattr(flwr_strategies, args.strategy)
    # strategy_class = patch_strategy(strategy_class)
    strategy_instance = strategy_class(min_fit_clients=args.min_clients, min_available_clients=args.available_clients, fraction_fit=args.fraction_fit, evaluate_fn=evaluate, **args.strategy_config)
    log(INFO, f"Using strategy: {args.strategy}")
    log(INFO, f"Strategy parameters: {args.strategy_config}")
    start_fl_server(strategy=strategy_instance, rounds=args.number_of_rounds)
    end_time = datetime.datetime.now()
    duration = end_time-start_time
    log(INFO, f"{end_time}: Server script finished, took {duration.total_seconds()} seconds")