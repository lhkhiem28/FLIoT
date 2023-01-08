# fliot

## Setup

1. Clone this repo
2. Create a new folder with the name datasets
3. Download the dataset for experiments from [here](https://drive.google.com/drive/folders/1JqMeujxMVhY7eQD-JqGDRGvuERlHl8n6?usp=sharing), then extract it to the datasets folder

    The workspace structure must be like this: 
    ```bash
    .
    ├── datasets
    │   ├── CIFAR10-64-IID
    │   │   ├── clients
    │   │   │   ├── c0
    │   │   │   ├── c1
    │   │   │   └── ...
    │   │   └── test
    │   │       ├── 0
    │   │       ├── 1
    │   │       └── ...
    │   └── CIFAR10-64-IID.zip
    └── fliot
        ├── README.md
        └── source
            ├── data.py
            ├── engines.py
            ├── libs.py
            ├── main
            │   ├── client.py
            │   └── server.py
            ├── models
            │   └── cnn.py
            ├── partition.py
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
`python3 server.py --dataset="CIFAR10-64-IID" --num_classes=10 --project="Nano-8-IID-CIFAR10"`

On the client `i`:</br>
`cd fliot/source/main`</br>
`python3 client.py --dataset="CIFAR10-64-IID" --num_classes=10 --project="Nano-8-IID-CIFAR10" --client_id=i`

Other arguments that can be can change when needed:</br>
`--server_address`: IP address of the server; `--server_port`: Opened port on the server</br>
`--num_rounds`: Number of communication rounds</br>
`--num_clients`: Number of participating clients</br>