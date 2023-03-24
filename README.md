# fliot

## Setup

1. Clone this repo
2. Create a new folder with the name datasets
3. Download the dataset for experiments from [here](https://drive.google.com/drive/folders/13h-oYVRnR6Bz52JVa23O5_1jP6UAfQOO?usp=sharing), then extract it to the datasets folder

    The workspace structure must be like this:
    ```bash
    .
    ├── datasets
    │   ├── CIFAR10-64-IID
    │   │   ├── clients
    │   │   │   ├── c0
    │   │   │   │   ├── fit
    │   │   │   │   │   ├── 0
    │   │   │   │   │   ├── 1
    │   │   │   │   │   ├── .
    │   │   │   │   │   └── 9
    │   │   │   │   └── evaluate
    │   │   │   │       ├── 0
    │   │   │   │       ├── 1
    │   │   │   │       ├── .
    │   │   │   │       └── 9
    │   │   │   ├── ...
    │   │   │   └── c63
    │   │   └── test
    │   │       ├── 0
    │   │       ├── 1
    │   │       ├── .
    │   │       └── 9
    │   └── CIFAR10-64-IID.zip
    └── fliot
        ├── README.md
        └── source
            ├── data.py
            ├── engines.py
            ├── libs.py
            ├── partition.py
            ├── main
            │   ├── client.py
            │   └── server.py
            ├── models
            │   └── cnn.py
            └── strategies.py
    ```

4. Pull docker images from docker hub
- For Raspberry Pi devices:
    - `docker pull lhkhiem28/rpi:2.2`
- For Jetson devices:
    - `docker pull lhkhiem28/jetson:2.2.2`
5. Run docker images and execute into containers

## Quickstart

On the server:</br>
`cd fliot/source/main`</br>
`python3 server.py --dataset="CIFAR10-64-IID" --num_classes=10 --num_clients=8 --wandb_project=""`

On the client `i`:</br>
`cd fliot/source/main`</br>
`python3 client.py --dataset="CIFAR10-64-IID" --num_classes=10 --wandb_project="" --client_id=i`

Other arguments that can be can change when needed:</br>
`--server_address`:IP address of the server, `--server_port`:Opened port on the server</br>
`--num_rounds`:Number of communication rounds</br>