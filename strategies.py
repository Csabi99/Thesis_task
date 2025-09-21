from flwr.server.strategy import *
from flwr.common import (
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    GetParametersIns,
    Code,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    FitIns,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from typing import Dict, Optional, Callable, Tuple
from logging import INFO, WARN, ERROR
from flwr.server.client_proxy import ClientProxy
import numpy as np
from typing import List, Union
import torch
import torch.nn as nn
from io import BytesIO


class FedAvgMDefault(FedAvgM):
    """Federated Averaging with Momentum strategy.

    Implementation based on https://arxiv.org/abs/1909.06335

    Parameters
    ----------
    fraction_fit : float, optional
        Fraction of clients used during training. Defaults to 1.0.
    fraction_evaluate : float, optional
        Fraction of clients used during validation. Defaults to 1.0.
    min_fit_clients : int, optional
        Minimum number of clients used during training. Defaults to 2.
    min_evaluate_clients : int, optional
        Minimum number of clients used during validation. Defaults to 2.
    min_available_clients : int, optional
        Minimum number of total clients in the system. Defaults to 2.
    evaluate_fn : Optional[Callable[[int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]]
        Optional function used for validation. Defaults to None.
    on_fit_config_fn : Callable[[int], Dict[str, Scalar]], optional
        Function used to configure training. Defaults to None.
    on_evaluate_config_fn : Callable[[int], Dict[str, Scalar]], optional
        Function used to configure validation. Defaults to None.
    accept_failures : bool, optional
        Whether or not accept rounds containing failures. Defaults to True.
    initial_parameters : Parameters, optional
        Initial global model parameters.
    server_learning_rate: float
        Server-side learning rate used in server-side optimization.
        Defaults to 1.0.
    server_momentum: float
        Server-side momentum factor used for FedAvgM. Defaults to 0.0.
    """

    # pylint: disable=too-many-arguments,too-many-instance-attributes, line-too-long
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        server_learning_rate: float = 1.0,
        server_momentum: float = 0.0,
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            server_learning_rate=server_learning_rate,
            server_momentum=server_momentum
        )  
    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        """Initialize global model parameters."""
        random_client = client_manager.sample(1)[0]
        ins = GetParametersIns(config={})
        get_parameters_res = random_client.get_parameters(
            ins=ins, timeout=100000, group_id=0
        )
        if get_parameters_res.status.code == Code.OK:
            log(INFO, "Received initial parameters from one random client")
        else:
            log(
                WARN,
                "Failed to receive initial parameters from the client."
                " Empty initial parameters will be used.",
            )
        self.initial_parameters = get_parameters_res.parameters
        return self.initial_parameters   

    

class FedCM(FedAvg):
    """FedCM [2021] strategy.

    Federated Learning with Client-level Momentum.

    Paper: https://arxiv.org/pdf/2106.10874

    Parameters
    ----------
    fraction_fit : float, optional
        Fraction of clients used during training. Defaults to 1.0.
    fraction_evaluate : float, optional
        Fraction of clients used during validation. Defaults to 1.0.
    min_fit_clients : int, optional
        Minimum number of clients used during training. Defaults to 2.
    min_evaluate_clients : int, optional
        Minimum number of clients used during validation. Defaults to 2.
    min_available_clients : int, optional
        Minimum number of total clients in the system. Defaults to 2.
    evaluate_fn : Optional[Callable[[int, NDArrays, Dict[str, Scalar]],Optional[Tuple[float, Dict[str, Scalar]]]]]
        Optional function used for validation. Defaults to None.
    on_fit_config_fn : Callable[[int], Dict[str, Scalar]], optional
        Function used to configure training. Defaults to None.
    on_evaluate_config_fn : Callable[[int], Dict[str, Scalar]], optional
        Function used to configure validation. Defaults to None.
    accept_failures : bool, optional
        Whether or not accept rounds containing failures. Defaults to True.
    initial_parameters : Parameters
        Initial global model parameters.
    fit_metrics_aggregation_fn : Optional[MetricsAggregationFn]
        Metrics aggregation function, optional.
    evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn]
        Metrics aggregation function, optional.
    eta : float, optional
        Server-side learning rate. Defaults to 1e-1.
    eta_l : float, optional
        Client-side learning rate. Defaults to 1e-1.
    epoch_l : int, optional
        Number of local epochs at client. Defaults to 3.
    decay_alfa : float, optional
        Decay factor for the momentum at client. Defaults to 1e-1.
    """

    # pylint: disable=too-many-arguments,too-many-instance-attributes,too-many-locals
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        eta: float = 1e-1,
        eta_l: float = 1e-1,
        epoch_l: int = 3,
        decay_alfa : float = 1e-1,
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        self.eta = eta
        self.eta_l = eta_l
        self.epoch_l = epoch_l
        self.decay_alfa = decay_alfa
        self.delta_t = None
        self.current_weights = None
        self.params = None
        self.optimizer = None
        if on_fit_config_fn is None:
            self.on_fit_config_fn = self.default_on_fit_config_fn

    def default_on_fit_config_fn(self, server_round: int) -> Dict[str, Scalar]:
        """Default configuration function for training."""
        grad = self.delta_t is not None # client should take the last grad_length number of elements in the param array
        mode = "FedCM"
        config = {"epoch": self.epoch_l, "mode": mode, "grad": grad, "decay_alfa": self.decay_alfa}
        return config
    
    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        """Initialize global model parameters."""
        log(INFO, "Requesting initial parameters from one random client")
        random_client = client_manager.sample(1)[0]
        ins = GetParametersIns(config={})
        get_parameters_res = random_client.get_parameters(
            ins=ins, timeout=100000, group_id=0
        )
        if get_parameters_res.status.code == Code.OK:
            log(INFO, "Received initial parameters from one random client")
        else:
            log(
                ERROR,
                "Failed to receive initial parameters from the client."
                "Can't proceed without initial params. Exiting...",
            )
            exit()
        self.initial_parameters = get_parameters_res.parameters
        return self.initial_parameters   

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"FedCM(accept_failures={self.accept_failures})"
        return rep

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        model_params = parameters_to_ndarrays(parameters)
        np.save("/app/logs/model_params.npy", np.array(model_params, dtype=object))

        gradient = self.delta_t if self.delta_t is not None else np.array([np.zeros_like(param_array) for param_array in model_params], dtype=object)
        np.save("/app/logs/gradient.npy", np.array(gradient, dtype=object))
        combined_arrays = np.array([*model_params, *gradient], dtype=object)
        np.save("/app/logs/combined.npy", np.array(combined_arrays, dtype=object))
        end_parameters = ndarrays_to_parameters(combined_arrays)
        config = {**config, "grad_length": len(gradient)}
        fit_ins = FitIns(end_parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]    
        

def aggregate_fit(self, server_round, results, failures):
    if not results:
        return None, {}
    # Do not aggregate if there are failures and failures are not accepted
    if not self.accept_failures and failures:
        return None, {}

    if self.current_weights is None:
        if self.initial_parameters is not None:
            self.current_weights = self.initial_parameters
        else:
            log(ERROR, "No initial weights. Can't proceed. Exiting...")
            return None, {}

    current_weights = parameters_to_ndarrays(self.current_weights)

    # Get client deltas: Δ_i = x_i,K - x_t
    deltas = []
    for _, fit_res in results:
        client_weights = parameters_to_ndarrays(fit_res.parameters)
        delta = [cw - gw for cw, gw in zip(client_weights, current_weights)]
        deltas.append(delta)

    # Average client updates
    num_clients = len(deltas)
    sum_delta = [
        sum(d[i] for d in deltas)
        for i in range(len(deltas[0]))
    ]

    # Correct scaling
    scale = -1.0 / (self.eta_l * self.epoch_l * num_clients)

    # Now apply scale
    delta_t_plus_1 = [scale * d for d in sum_delta]

    # # Update Δ_t with controlled momentum
    # if self.delta_t is None:
    #     self.delta_t = [np.zeros_like(d) for d in scaled_delta]

    # self.delta_t = [
    #     self.alpha * old + (1 - self.alpha) * new
    #     for old, new in zip(self.delta_t, scaled_delta)
    # ]

    # Update global weights: x_{t+1} = x_t - η_g * Δ_{t+1}
    new_weights = [
        w - self.eta * d
        for w, d in zip(current_weights, delta_t_plus_1)
    ]

    self.current_weights = ndarrays_to_parameters(new_weights)
    return self.current_weights, {}


#TODO (here based on fedadam but can use fedavgm as well)
    # def aggregate_fit(
    #     self,
    #     server_round: int,
    #     results: List[Tuple[ClientProxy, FitRes]],
    #     failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    # ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
    #     """Aggregate fit results using weighted average."""
    #     log(INFO, "Aggreagete fit started")
    #     # log(INFO, f"Server round: {server_round}")
    #     # log(INFO, f"Results: {'-----'.join([str(fr) for cp, fr in results])}")    
    #     fedavg_parameters_aggregated, metrics_aggregated = super().aggregate_fit(
    #         server_round=server_round, results=results, failures=failures
    #     )
    #     if fedavg_parameters_aggregated is None:
    #         log(INFO, "No parameters aggregated")
    #         return None, {}
    #     if self.current_weights is None:
    #         if self.initial_parameters is not None:
    #             self.current_weights = self.initial_parameters
    #         else:
    #             log(ERROR, "No initial parameters set. Can't proceed. Exiting...")
    #             exit()

    #     fedavg_weights_aggregate = parameters_to_ndarrays(fedavg_parameters_aggregated)
    #     current_weights = parameters_to_ndarrays(self.current_weights)
    #     # line 11. dti
    #     self.delta_t: NDArrays = [
    #         (x - y)/(self.eta_l*self.epoch_l*(-1)) for x, y in zip(fedavg_weights_aggregate, current_weights)
    #     ]
    #     #dt average computation is missing in the original implementation!! should be added here (lien 13) -> d_t_next (actually it was done with weight aggregate)
    #     # this gradient should be sent to the clients so they can use it to update their local momentum vector

    #     new_weights = [
    #         x - self.eta * y
    #         for x, y in zip(current_weights, self.delta_t)
    #     ]
    #     self.current_weights = ndarrays_to_parameters(new_weights)

    #     return self.current_weights, metrics_aggregated

    # def aggregate_fit(
    #     self,
    #     server_round: int,
    #     results: List[Tuple[ClientProxy, FitRes]],
    #     failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    # ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
    #     """Aggregate fit results using weighted average."""
    #     log(INFO, "Aggreagete fit started")
    #     # log(INFO, f"Server round: {server_round}")
    #     # log(INFO, f"Results: {'-----'.join([str(fr) for cp, fr in results])}")    
    #     fedavg_parameters_aggregated, metrics_aggregated = super().aggregate_fit(
    #         server_round=server_round, results=results, failures=failures
    #     )
    #     if fedavg_parameters_aggregated is None:
    #         log(INFO, "No parameters aggregated")
    #         return None, {}
    #     fedavg_weights_aggregate = parameters_to_ndarrays(fedavg_parameters_aggregated)
    #     if self.params is None:
    #         # Initialize torch Parameters from initial server weights
    #         initial_weights = parameters_to_ndarrays(self.initial_parameters)
    #         self.params = nn.ParameterList([
    #             nn.Parameter(torch.tensor(w, dtype=torch.float32), requires_grad=True)
    #             for w in initial_weights
    #         ])
    #         self.optimizer = torch.optim.Adam(self.params)
    #     with torch.no_grad():
    #         for p, new_w in zip(self.params, fedavg_weights_aggregate):
    #             p.grad = p.data - torch.tensor(new_w, dtype=torch.float32)

    #     self.optimizer.step()
    #     self.optimizer.zero_grad()
        
    #     updated_weights = [p.detach().cpu().numpy() for p in self.params]
    #     return ndarrays_to_parameters(updated_weights), metrics_aggregated
                   

class FedAdamDefault(FedAdam):
    """FedAdam - Adaptive Federated Optimization using Adam.
    Customized: initial_parameters is only optional. If it is None, initialize_parameters() function will set the initial_parameters.

    Implementation based on https://arxiv.org/abs/2003.00295v5

    Parameters
    ----------
    fraction_fit : float, optional
        Fraction of clients used during training. Defaults to 1.0.
    fraction_evaluate : float, optional
        Fraction of clients used during validation. Defaults to 1.0.
    min_fit_clients : int, optional
        Minimum number of clients used during training. Defaults to 2.
    min_evaluate_clients : int, optional
        Minimum number of clients used during validation. Defaults to 2.
    min_available_clients : int, optional
        Minimum number of total clients in the system. Defaults to 2.
    evaluate_fn : Optional[Callable[[int, NDArrays, Dict[str, Scalar]],Optional[Tuple[float, Dict[str, Scalar]]]]]
        Optional function used for validation. Defaults to None.
    on_fit_config_fn : Callable[[int], Dict[str, Scalar]], optional
        Function used to configure training. Defaults to None.
    on_evaluate_config_fn : Callable[[int], Dict[str, Scalar]], optional
        Function used to configure validation. Defaults to None.
    accept_failures : bool, optional
        Whether or not accept rounds containing failures. Defaults to True.
    initial_parameters : Parameters
        Initial global model parameters.
    fit_metrics_aggregation_fn : Optional[MetricsAggregationFn]
        Metrics aggregation function, optional.
    evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn]
        Metrics aggregation function, optional.
    eta : float, optional
        Server-side learning rate. Defaults to 1e-1.
    eta_l : float, optional
        Client-side learning rate. Defaults to 1e-1.
    beta_1 : float, optional
        Momentum parameter. Defaults to 0.9.
    beta_2 : float, optional
        Second moment parameter. Defaults to 0.99.
    tau : float, optional
        Controls the algorithm's degree of adaptability. Defaults to 1e-9.
    """

    # pylint: disable=too-many-arguments,too-many-instance-attributes,too-many-locals
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Parameters = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        eta: float = 1e-1,
        eta_l: float = 1e-1,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        tau: float = 1e-9,
    ) -> None:
        if initial_parameters is None:
            arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            buffer = BytesIO()
            np.save(buffer, arr, allow_pickle=False)
            serialized = buffer.getvalue()
            params = Parameters(tensors=[serialized], tensor_type="numpy")
            initial_parameters = params
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            eta=eta,
            eta_l=eta_l,
            beta_1=beta_1,
            beta_2=beta_2,
            tau=tau,
        )

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"FedAdam(accept_failures={self.accept_failures})"
        return rep
    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        """Initialize global model parameters."""
        random_client = client_manager.sample(1)[0]
        ins = GetParametersIns(config={})
        get_parameters_res = random_client.get_parameters(
            ins=ins, timeout=100000, group_id=0
        )
        if get_parameters_res.status.code == Code.OK:
            log(INFO, "Received initial parameters from one random client")
        else:
            log(
                WARN,
                "Failed to receive initial parameters from the client."
                " Empty initial parameters will be used.",
            )
        self.initial_parameters = get_parameters_res.parameters
        self.current_weights = parameters_to_ndarrays(self.initial_parameters)
        log(INFO, "Types of weights:")
        for x in self.current_weights:
            log(INFO, f"Layer : x={x.dtype}")
        return self.initial_parameters
  
    
class FedAdagradDefault(FedAdagrad):
    """FedAdagrad strategy - Adaptive Federated Optimization using Adagrad.

    Implementation based on https://arxiv.org/abs/2003.00295v5

    Parameters
    ----------
    fraction_fit : float, optional
        Fraction of clients used during training. Defaults to 1.0.
    fraction_evaluate : float, optional
        Fraction of clients used during validation. Defaults to 1.0.
    min_fit_clients : int, optional
        Minimum number of clients used during training. Defaults to 2.
    min_evaluate_clients : int, optional
        Minimum number of clients used during validation. Defaults to 2.
    min_available_clients : int, optional
        Minimum number of total clients in the system. Defaults to 2.
    evaluate_fn : Optional[Callable[[int, NDArrays, Dict[str, Scalar]],Optional[Tuple[float, Dict[str, Scalar]]]]]
        Optional function used for validation. Defaults to None.
    on_fit_config_fn : Callable[[int], Dict[str, Scalar]], optional
        Function used to configure training. Defaults to None.
    on_evaluate_config_fn : Callable[[int], Dict[str, Scalar]], optional
        Function used to configure validation. Defaults to None.
    fit_metrics_aggregation_fn : Optional[MetricsAggregationFn]
        Metrics aggregation function, optional.
    evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn]
        Metrics aggregation function, optional.
    accept_failures : bool, optional
        Whether or not accept rounds containing failures. Defaults to True.
    initial_parameters : Parameters
        Initial global model parameters.
    eta : float, optional
        Server-side learning rate. Defaults to 1e-1.
    eta_l : float, optional
        Client-side learning rate. Defaults to 1e-1.
    tau : float, optional
        Controls the algorithm's degree of adaptability. Defaults to 1e-9.
    """

    # pylint: disable=too-many-arguments,too-many-locals,too-many-instance-attributes
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        accept_failures: bool = True,
        initial_parameters: Parameters = None,
        eta: float = 1e-1,
        eta_l: float = 1e-1,
        tau: float = 1e-9,
    ) -> None:
        if initial_parameters is None:
            arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            buffer = BytesIO()
            np.save(buffer, arr, allow_pickle=False)
            serialized = buffer.getvalue()
            params = Parameters(tensors=[serialized], tensor_type="numpy")
            initial_parameters = params        
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            eta=eta,
            eta_l=eta_l,
            tau=tau,
        )
    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"FedAdagrad(accept_failures={self.accept_failures})"
        return rep
    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        """Initialize global model parameters."""
        random_client = client_manager.sample(1)[0]
        ins = GetParametersIns(config={})
        get_parameters_res = random_client.get_parameters(
            ins=ins, timeout=100000, group_id=0
        )
        if get_parameters_res.status.code == Code.OK:
            log(INFO, "Received initial parameters from one random client")
        else:
            log(
                WARN,
                "Failed to receive initial parameters from the client."
                " Empty initial parameters will be used.",
            )
        self.initial_parameters = get_parameters_res.parameters
        self.current_weights = parameters_to_ndarrays(self.initial_parameters)
        log(INFO, "Types of weights:")
        for x in self.current_weights:
            log(INFO, f"Layer : x={x.dtype}")
        return self.initial_parameters     
  

class FedYogiDefault(FedYogi):
    """FedYogi [Reddi et al., 2020] strategy.

    Implementation based on https://arxiv.org/abs/2003.00295v5

    Parameters
    ----------
    fraction_fit : float, optional
        Fraction of clients used during training. Defaults to 1.0.
    fraction_evaluate : float, optional
        Fraction of clients used during validation. Defaults to 1.0.
    min_fit_clients : int, optional
        Minimum number of clients used during training. Defaults to 2.
    min_evaluate_clients : int, optional
        Minimum number of clients used during validation. Defaults to 2.
    min_available_clients : int, optional
        Minimum number of total clients in the system. Defaults to 2.
    evaluate_fn : Optional[Callable[[int, NDArrays, Dict[str, Scalar]], Optional[
    Tuple[float, Dict[str, Scalar]]]]]
        Optional function used for validation. Defaults to None.
    on_fit_config_fn : Callable[[int], Dict[str, Scalar]], optional
        Function used to configure training. Defaults to None.
    on_evaluate_config_fn : Callable[[int], Dict[str, Scalar]], optional
        Function used to configure validation. Defaults to None.
    accept_failures : bool, optional
        Whether or not accept rounds containing failures. Defaults to True.
    initial_parameters : Parameters
        Initial global model parameters.
    fit_metrics_aggregation_fn : Optional[MetricsAggregationFn]
        Metrics aggregation function, optional.
    evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn]
        Metrics aggregation function, optional.
    eta : float, optional
        Server-side learning rate. Defaults to 1e-2.
    eta_l : float, optional
        Client-side learning rate. Defaults to 0.0316.
    beta_1 : float, optional
        Momentum parameter. Defaults to 0.9.
    beta_2 : float, optional
        Second moment parameter. Defaults to 0.99.
    tau : float, optional
        Controls the algorithm's degree of adaptability.
        Defaults to 1e-3.
    """

    # pylint: disable=too-many-arguments,too-many-locals,too-many-instance-attributes
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, dict[str, Scalar]],
                Optional[tuple[float, dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Parameters = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        eta: float = 1e-2,
        eta_l: float = 0.0316,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        tau: float = 1e-3,
    ) -> None:
        if initial_parameters is None:
            arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            buffer = BytesIO()
            np.save(buffer, arr, allow_pickle=False)
            serialized = buffer.getvalue()
            params = Parameters(tensors=[serialized], tensor_type="numpy")
            initial_parameters = params            
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            eta=eta,
            eta_l=eta_l,
            beta_1=beta_1,
            beta_2=beta_2,
            tau=tau,
        )
    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"FedAdagrad(accept_failures={self.accept_failures})"
        return rep
    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        """Initialize global model parameters."""
        random_client = client_manager.sample(1)[0]
        ins = GetParametersIns(config={})
        get_parameters_res = random_client.get_parameters(
            ins=ins, timeout=100000, group_id=0
        )
        if get_parameters_res.status.code == Code.OK:
            log(INFO, "Received initial parameters from one random client")
        else:
            log(
                WARN,
                "Failed to receive initial parameters from the client."
                " Empty initial parameters will be used.",
            )
        self.initial_parameters = get_parameters_res.parameters
        self.current_weights = parameters_to_ndarrays(self.initial_parameters)
        log(INFO, "Types of weights:")
        for x in self.current_weights:
            log(INFO, f"Layer : x={x.dtype}")
        return self.initial_parameters                  