import os, sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__), sys.path.append(os.path.abspath(os.path.join(__dir__, "..")))
from libs import *

from strategies import *
from data import ImageDataset

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
        project = args.dataset.split("/")[0], name = "num_rounds = {:3}, num_epochs = {:3}".format(args.num_rounds, args.num_epochs), 
        # mode = "disabled", 
    )

    test_loaders = {
        "test":torch.utils.data.DataLoader(
            ImageDataset(
                data_dir = "../../datasets/{}/{}/".format(args.dataset.split("/")[0], "server"), 
            ), 
            num_workers = 0, batch_size = 32, 
            shuffle = False, 
        )
    }
    server_model = torchvision.models.efficientnet_b2()
    server_model.classifier[1] = nn.Linear(
        server_model.classifier[1].in_features, args.num_classes, 
    )
    flwr.server.start_server(
        server_address = "{}:{}".format(args.server_address, args.server_port), 
        config = flwr.server.ServerConfig(num_rounds = args.num_rounds), 
        strategy = FedAvg(
            min_available_clients = args.num_clients, min_fit_clients = args.num_clients, 
            test_loaders = test_loaders, 
            server_model = server_model, 
        ), 
    )

    wandb.finish()