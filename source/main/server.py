
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
    parser.add_argument("--server_address", type = str, default = "127.0.0.1"), parser.add_argument("--server_port", type = int, default = 9999)
    parser.add_argument("--num_rounds", type = int, default = 100)
    parser.add_argument("--num_clients", type = int, default = 8)
    parser.add_argument("--dataset", type = str), parser.add_argument("--num_classes", type = int)
    parser.add_argument("--project", type = str)
    args = parser.parse_args()
    wandb.login()
    wandb.init(
        entity = "fliot", project = args.project, 
        name = "server", 
    )

    if "MNIST" in args.dataset:
        image_size, num_channels,  = 28, 1, 
    else:
        image_size, num_channels,  = 32, 3, 

    server_model = CNN3(
        image_size, num_channels, 
        num_classes = 10, 
    )
    initial_parameters = [value.cpu().numpy() for key, value in server_model.state_dict().items()]
    initial_parameters = fl.common.ndarrays_to_parameters(initial_parameters)
    save_ckp_dir = "../../ckps/{}".format(args.project)
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
            num_channels = num_channels, 
        ), 
        batch_size = 32, 
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