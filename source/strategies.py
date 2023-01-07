
import os, sys
from libs import *

def metrics_aggregation_fn(results):
    fit_losses, fit_accuracies,  = [result["fit_loss"]*num_examples for num_examples, result in results], [result["fit_accuracy"]*num_examples for num_examples, result in results], 
    evaluate_losses, evaluate_accuracies,  = [result["evaluate_loss"]*num_examples for num_examples, result in results], [result["evaluate_accuracy"]*num_examples for num_examples, result in results], 

    sum_examples = sum([num_examples for num_examples, result in results])
    aggregated_metrics = {
        "fit_loss":sum(fit_losses)/sum_examples, "fit_accuracy":sum(fit_accuracies)/sum_examples, 
        "evaluate_loss":sum(evaluate_losses)/sum_examples, "evaluate_accuracy":sum(evaluate_accuracies)/sum_examples, 
    }

    return aggregated_metrics

class FedAvg(fl.server.strategy.FedAvg):
    def __init__(self, 
        server_model, 
        save_ckp_dir, 
        *args, **kwargs
    ):
        self.server_model = server_model
        self.save_ckp_dir = save_ckp_dir
        super().__init__(*args, **kwargs)

        self.server_accuracy = 0.0

    def aggregate_fit(self, 
        server_round, 
        results, failures, 
    ):
        aggregated_metrics = metrics_aggregation_fn([(result.num_examples, result.metrics) for _, result in results])
        wandb.log({"evaluate_loss":aggregated_metrics["evaluate_loss"], "evaluate_accuracy":aggregated_metrics["evaluate_accuracy"], }, step = server_round)
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
        if aggregated_metrics["evaluate_accuracy"] > self.server_accuracy:
            torch.save(
                self.server_model, 
                "{}/server.ptl".format(self.save_ckp_dir), 
            )
            self.server_accuracy = aggregated_metrics["evaluate_accuracy"]

        aggregated_parameters = [value.cpu().numpy() for key, value in self.server_model.state_dict().items()]
        aggregated_parameters = fl.common.ndarrays_to_parameters(aggregated_parameters)

        return aggregated_parameters, {}