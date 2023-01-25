import os, sys
from libs import *

from engines import *

class FedAvg(fl.server.strategy.FedAvg):
    def __init__(self, 
        test_loader, 
        server_model, 
        *args, **kwargs, 
    ):
        self.test_loader = test_loader
        self.server_model = server_model
        super().__init__(*args, **kwargs, )

        self.server_accuracy = 0.0

    def aggregate_fit(self, 
        server_round, 
        results, failures, 
    ):
        aggregated_parameters = super().aggregate_fit(
            server_round, 
            results, failures, 
        )[0]
        aggregated_parameters = fl.common.parameters_to_ndarrays(aggregated_parameters)
        aggregated_keys = [key for key in self.server_model.state_dict().keys()]
        self.server_model.load_state_dict(
            collections.OrderedDict({key:torch.tensor(value) for key, value in zip(aggregated_keys, aggregated_parameters)}), 
            strict = False, 
        )

        results = server_test_fn(
            self.test_loader, 
            self.server_model, 
            device = torch.device("cpu"), 
        )
        wandb.log({"test_loss":results["test_loss"], "test_accuracy":results["test_accuracy"], }, step = server_round)

        aggregated_parameters = [value.cpu().numpy() for key, value in self.server_model.state_dict().items()]
        aggregated_parameters = fl.common.ndarrays_to_parameters(aggregated_parameters)

        return aggregated_parameters, {}