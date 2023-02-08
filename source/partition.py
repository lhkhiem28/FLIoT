import os, sys
from libs import *

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type = str), parser.add_argument("--num_classes", type = int)
parser.add_argument("--num_clients", type = int, default = 64)
parser.add_argument("--non_iid", action = "store_true"), parser.add_argument("--num_chunks", type = int)
args = parser.parse_args()

if not os.path.exists(args.data_dir + "clients/"):
    for c in range(args.num_clients):
        for cls in range(args.num_classes):
            os.makedirs(args.data_dir + "clients/c{}/fit/{}/".format(str(c), str(cls)))
            os.makedirs(args.data_dir + "clients/c{}/evaluate/{}/".format(str(c), str(cls)))

image_files = glob.glob(args.data_dir + "train/*/*")
if not args.non_iid:
    np.random.shuffle(image_files)
    num_examples_per_chunk = len(image_files) // args.num_clients
    chunks = list(range(args.num_clients))
    for c, chunk in enumerate(chunks):
        client_image_files = image_files[num_examples_per_chunk*(chunk):num_examples_per_chunk*(chunk + 1)]
        np.random.shuffle(client_image_files)
        for image_file in client_image_files[:int(len(client_image_files)*0.9)]:
            label = int(image_file.split("/")[-2])
            shutil.copy(image_file, args.data_dir + "clients/c{}/fit/{}/".format(str(c), str(label)))
        for image_file in client_image_files[int(len(client_image_files)*0.9):]:
            label = int(image_file.split("/")[-2])
            shutil.copy(image_file, args.data_dir + "clients/c{}/evaluate/{}/".format(str(c), str(label)))
else:
    image_files = [image_file for _, image_file in sorted(zip([int(image_file.split("/")[-2]) for image_file in image_files], image_files))]
    num_examples_per_chunk = len(image_files) // (args.num_clients*args.num_chunks)
    chunks = list(range(args.num_clients*args.num_chunks))
    np.random.shuffle(chunks)
    for c in range(args.num_clients):
        client_chunks = chunks[args.num_chunks*(c):args.num_chunks*(c + 1)]
        client_image_files = []
        for chunk in client_chunks:
            client_image_files += image_files[num_examples_per_chunk*(chunk):num_examples_per_chunk*(chunk + 1)]
        np.random.shuffle(client_image_files)
        for image_file in client_image_files[:int(len(client_image_files)*0.9)]:
            label = int(image_file.split("/")[-2])
            shutil.copy(image_file, args.data_dir + "clients/c{}/fit/{}/".format(str(c), str(label)))
        for image_file in client_image_files[int(len(client_image_files)*0.9):]:
            label = int(image_file.split("/")[-2])
            shutil.copy(image_file, args.data_dir + "clients/c{}/evaluate/{}/".format(str(c), str(label)))