import os, sys
from libs import *

from engines import server_test_fn

class FedAvg(flwr.server.strategy.FedAvg):
    def __init__(self, 
        test_loaders, 
        server_model, 
        *args, **kwargs
    ):
        self.test_loaders = test_loaders
        self.server_model = server_model
        super().__init__(*args, **kwargs)

    def aggregate_fit(self, 
        server_round, 
        results, failures, 
    ):
        aggregated_parameters = super().aggregate_fit(
            server_round, 
            results, failures, 
        )[0]
        aggregated_parameters = flwr.common.parameters_to_ndarrays(aggregated_parameters)

        aggregated_keys = [key for key in self.server_model.state_dict().keys()]
        self.server_model.load_state_dict(
            collections.OrderedDict({key:torch.tensor(value) for key, value in zip(aggregated_keys, aggregated_parameters)}), 
            strict = False, 
        )
        metrics = server_test_fn(
            self.test_loaders, 
            self.server_model, 
        )
        wandb.log(
            {
                "test_loss":metrics["test_loss"], "test_accuracy":metrics["test_accuracy"]
            }, 
            step = server_round, 
        )

        aggregated_parameters = [value.cpu().numpy() for key, value in self.server_model.state_dict().items()]
        aggregated_parameters = flwr.common.ndarrays_to_parameters(aggregated_parameters)

        return aggregated_parameters, {}