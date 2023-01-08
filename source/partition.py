
import os, sys
from libs import *

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type = str), parser.add_argument("--num_classes", type = int)
parser.add_argument("--num_clients", type = int, default = 64)
args = parser.parse_args()

if not os.path.exists(args.data_dir + "clients/"):
    for c in range(args.num_clients):
        for label in range(args.num_classes):
            os.makedirs(args.data_dir + "clients/c{}/fit/{}/".format(str(c), str(label)))
            os.makedirs(args.data_dir + "clients/c{}/evaluate/{}/".format(str(c), str(label)))

image_files = glob.glob(args.data_dir + "train/*/*")
np.random.shuffle(image_files)

num_examples_per_client = len(image_files) // args.num_clients
for c in range(args.num_clients):
    client_image_files = image_files[num_examples_per_client*c:num_examples_per_client*(c + 1)]
    np.random.shuffle(client_image_files)
    for image_file in client_image_files[:int(num_examples_per_client*0.9)]:
        label = int(image_file.split("/")[-2])
        shutil.copy(image_file, args.data_dir + "clients/c{}/fit/{}/".format(str(c), str(label)))
    for image_file in client_image_files[int(num_examples_per_client*0.9):]:
        label = int(image_file.split("/")[-2])
        shutil.copy(image_file, args.data_dir + "clients/c{}/evaluate/{}/".format(str(c), str(label)))