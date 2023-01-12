
import os, sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__), sys.path.append(os.path.abspath(os.path.join(__dir__, "..")))
from libs import *

from strategies import FedAvg
from data import ImageDataset
from models.cnn import *
from engines import server_test_fn

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_address", type = str, default = "192.168.50.79"), parser.add_argument("--server_port", type = int, default = 9999)
    parser.add_argument("--num_rounds", type = int, default = 300)
    parser.add_argument("--num_clients", type = int, default = 8)
    parser.add_argument("--dataset", type = str), parser.add_argument("--num_classes", type = int)
    parser.add_argument("--wandb_key", type = str, default = "3304b9a0c28f65f7d1097ef922eca22b370116cb")
    parser.add_argument("--wandb_entity", type = str, default = "fliot"), parser.add_argument("--wandb_project", type = str), 
    args = parser.parse_args()
    wandb.login(key = args.wandb_key)
    wandb.init(
        entity = args.wandb_entity, project = args.wandb_project, 
        name = "server", 
    )

    if "MNIST" in args.dataset:
        image_size, num_channels,  = 28, 1, 
    else:
        image_size, num_channels,  = 32, 3, 

    server_model = CNN3(
        image_size, num_channels, 
        num_classes = args.num_classes, 
    )
    initial_parameters = [value.cpu().numpy() for key, value in server_model.state_dict().items()]
    initial_parameters = fl.common.ndarrays_to_parameters(initial_parameters)
    save_ckp_dir = "../../ckps/{}".format(args.wandb_project)
    if not os.path.exists(save_ckp_dir):
        os.makedirs(save_ckp_dir)
    fl.server.start_server(
        server_address = "{}:{}".format(args.server_address, args.server_port), 
        config = fl.server.ServerConfig(
            num_rounds = args.num_rounds, 
        ), 
        strategy = FedAvg(
            min_available_clients = args.num_clients, min_fit_clients = args.num_clients, 
            server_model = server_model, 
            initial_parameters = initial_parameters, 
            save_ckp_dir = save_ckp_dir, 
        )
    )

    test_loader = torch.utils.data.DataLoader(
        ImageDataset(
            data_dir = "../../../datasets/{}/test/".format(args.dataset), 
            image_size = image_size, 
        ), 
        batch_size = 16, 
    )
    server_model = torch.load(
        "{}/server.ptl".format(save_ckp_dir), 
        map_location = "cpu", 
    )
    results = server_test_fn(
        test_loader, 
        server_model, 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
    )
    mlogger = open("{}/server.txt".format(save_ckp_dir), "a")
    mlogger.write("{:<8} - loss:{:.4f}, accuracy:{:.4f}".format(
        "test", 
        results["test_loss"], results["test_accuracy"], 
    ))