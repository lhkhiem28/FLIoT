import os, sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__), sys.path.append(os.path.abspath(os.path.join(__dir__, "..")))
from libs import *

from strategies import *
from data import ImageDataset
from engines import client_fit_fn

class Client(flwr.client.NumPyClient):
    def __init__(self, 
        fit_loaders, num_epochs, 
        client_model, 
        client_optim, 
    ):
        self.fit_loaders, self.num_epochs,  = fit_loaders, num_epochs, 
        self.client_model = client_model
        self.client_optim = client_optim

    def get_parameters(self, 
        config, 
    ):
        return [value.cpu().numpy() for key, value in self.client_model.state_dict().items()]

    def fit(self, 
        parameters, config, 
    ):
        keys = [key for key in self.client_model.state_dict().keys()]
        self.client_model.load_state_dict(
            collections.OrderedDict({key:torch.tensor(value) for key, value in zip(keys, parameters)}), 
            strict = False, 
        )
        metrics = client_fit_fn(
            self.fit_loaders, self.num_epochs, 
            self.client_model, 
            self.client_optim, 
        )

        return self.get_parameters({}), len(self.fit_loaders["fit"].dataset), metrics

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

    fit_loaders = {
        "fit":torch.utils.data.DataLoader(
            ImageDataset(
                data_dir = "../../datasets/{}/{}/".format(args.dataset, args.client_dataset), 
            ), 
            num_workers = 0, batch_size = 32, 
            shuffle = True, 
        )
    }
    client_model = torchvision.models.efficientnet_b2()
    client_model.classifier = nn.Linear(
        client_model.classifier.in_features, args.num_classes, 
    )
    client_optim = optim.Adam(
        client_model.parameters(), weight_decay = 5e-5, 
        lr = 1e-3, 
    )

    client = Client(
        fit_loaders, args.num_epochs, 
        client_model, 
        client_optim, 
    )
    flwr.client.start_numpy_client(
        server_address = "{}:{}".format(args.server_address, args.server_port), 
        client = client, 
    )