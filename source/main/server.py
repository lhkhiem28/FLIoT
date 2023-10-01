import os, sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__), sys.path.append(os.path.abspath(os.path.join(__dir__, "..")))
from libs import *

from strategies import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_address", type = str, default = "127.0.0.1"), parser.add_argument("--server_port", type = int, default = 9999)
    parser.add_argument("--dataset", type = str, default = "CIFAR10/clients-5.0"), parser.add_argument("--client_dataset", type = int)
    parser.add_argument("--num_classes", type = int, default = 10)
    parser.add_argument("--num_clients", type = int, default = 10)
    parser.add_argument("--num_rounds", type = int, default = 500)
    parser.add_argument("--num_epochs", type = int, default = 1)
    parser.add_argument("--wandb_entity", type = str, default = "khiemlhfx")
    args = parser.parse_args()
    wandb.login(key = "b8731afd2f2cefd26df285f59339b7834d05339b")
    wandb.init(
        entity = args.wandb_entity, project = args.dataset, 
        name = "num_rounds = {:3}, num_epochs = {:3}".format(args.num_rounds, args.num_epochs), 
    )

    server_model = torchvision.models.efficientnet_b2()
    server_model.classifier = nn.Linear(
        server_model.classifier.in_features, args.num_classes, 
    )
    initial_parameters = [value.cpu().numpy() for key, value in server_model.state_dict().items()]
    initial_parameters = flwr.common.ndarrays_to_parameters(initial_parameters)
    flwr.server.start_server(
        server_address = "{}:{}".format(args.server_address, args.server_port), 
        config = flwr.server.ServerConfig(num_rounds = args.num_rounds), 
        strategy = FedAvg(
            min_available_clients = args.num_clients, min_fit_clients = args.num_clients, 
            initial_parameters = initial_parameters, 
        ), 
    )

    wandb.finish()