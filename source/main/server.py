import os, sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__), sys.path.append(os.path.abspath(os.path.join(__dir__, "..")))
from libs import *

from strategies import FedAvg
from data import ImageDataset
from models.cnn import CNN3
from engines import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_address", type = str, default = "127.0.0.1"), parser.add_argument("--server_port", type = int, default = 9999)
    parser.add_argument("--dataset", type = str), parser.add_argument("--num_classes", type = int)
    parser.add_argument("--num_clients", type = int)
    parser.add_argument("--num_rounds", type = int, default = 500)
    parser.add_argument("--wandb_key", type = str, default = "5fc200276378a2457d9278516a0e8ca600914732")
    parser.add_argument("--wandb_entity", type = str, default = "lehuykhiem28011999"), parser.add_argument("--wandb_project", type = str), 
    args = parser.parse_args()
    wandb.login(key = args.wandb_key)
    wandb.init(
        entity = args.wandb_entity, project = args.wandb_project, 
        name = "server", 
    )

    test_loader = torch.utils.data.DataLoader(
        ImageDataset(
            data_dir = "../../../datasets/{}/test/".format(args.dataset), 
        ), 
        batch_size = 16, 
    )
    server_model = CNN3(
        num_classes = args.num_classes, 
    )
    initial_parameters = [value.cpu().numpy() for key, value in server_model.state_dict().items()]
    initial_parameters = fl.common.ndarrays_to_parameters(initial_parameters)
    fl.server.start_server(
        server_address = "{}:{}".format(args.server_address, args.server_port), 
        config = fl.server.ServerConfig(num_rounds = args.num_rounds), 
        strategy = FedAvg(
            min_available_clients = args.num_clients, min_fit_clients = args.num_clients, 
            test_loader = test_loader, 
            server_model = server_model, 
            initial_parameters = initial_parameters, 
        ), 
    )

    wandb.finish()