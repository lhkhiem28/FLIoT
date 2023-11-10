cd ../main/

nohup python server.py --server_port=9995 --dataset="CIFAR10/clients-5.0" --num_rounds=250 --num_epochs=2 > server.out &
sleep 30
nohup python client.py --server_port=9995 --dataset="CIFAR10/clients-5.0" --client_dataset=client00 --num_rounds=250 --num_epochs=2 > client00.out &
sleep 3
nohup python client.py --server_port=9995 --dataset="CIFAR10/clients-5.0" --client_dataset=client01 --num_rounds=250 --num_epochs=2 > client01.out &
sleep 3
nohup python client.py --server_port=9995 --dataset="CIFAR10/clients-5.0" --client_dataset=client02 --num_rounds=250 --num_epochs=2 > client02.out &
sleep 3
nohup python client.py --server_port=9995 --dataset="CIFAR10/clients-5.0" --client_dataset=client03 --num_rounds=250 --num_epochs=2 > client03.out &
sleep 3
nohup python client.py --server_port=9995 --dataset="CIFAR10/clients-5.0" --client_dataset=client04 --num_rounds=250 --num_epochs=2 > client04.out &
sleep 3
nohup python client.py --server_port=9995 --dataset="CIFAR10/clients-5.0" --client_dataset=client05 --num_rounds=250 --num_epochs=2 > client05.out &
sleep 3
nohup python client.py --server_port=9995 --dataset="CIFAR10/clients-5.0" --client_dataset=client06 --num_rounds=250 --num_epochs=2 > client06.out &
sleep 3
nohup python client.py --server_port=9995 --dataset="CIFAR10/clients-5.0" --client_dataset=client07 --num_rounds=250 --num_epochs=2 > client07.out &
sleep 3
nohup python client.py --server_port=9995 --dataset="CIFAR10/clients-5.0" --client_dataset=client08 --num_rounds=250 --num_epochs=2 > client08.out &
sleep 3
nohup python client.py --server_port=9995 --dataset="CIFAR10/clients-5.0" --client_dataset=client09 --num_rounds=250 --num_epochs=2 > client09.out &
sleep 3
nohup python client.py --server_port=9995 --dataset="CIFAR10/clients-5.0" --client_dataset=client10 --num_rounds=250 --num_epochs=2 > client10.out &
sleep 3
nohup python client.py --server_port=9995 --dataset="CIFAR10/clients-5.0" --client_dataset=client11 --num_rounds=250 --num_epochs=2 > client11.out &
sleep 3
nohup python client.py --server_port=9995 --dataset="CIFAR10/clients-5.0" --client_dataset=client12 --num_rounds=250 --num_epochs=2 > client12.out &
sleep 3
nohup python client.py --server_port=9995 --dataset="CIFAR10/clients-5.0" --client_dataset=client13 --num_rounds=250 --num_epochs=2 > client13.out &
sleep 3
nohup python client.py --server_port=9995 --dataset="CIFAR10/clients-5.0" --client_dataset=client14 --num_rounds=250 --num_epochs=2 > client14.out &
sleep 3
nohup python client.py --server_port=9995 --dataset="CIFAR10/clients-5.0" --client_dataset=client15 --num_rounds=250 --num_epochs=2 > client15.out &
sleep 3
nohup python client.py --server_port=9995 --dataset="CIFAR10/clients-5.0" --client_dataset=client16 --num_rounds=250 --num_epochs=2 > client16.out &
sleep 3
nohup python client.py --server_port=9995 --dataset="CIFAR10/clients-5.0" --client_dataset=client17 --num_rounds=250 --num_epochs=2 > client17.out &
sleep 3
nohup python client.py --server_port=9995 --dataset="CIFAR10/clients-5.0" --client_dataset=client18 --num_rounds=250 --num_epochs=2 > client18.out &
sleep 3
nohup python client.py --server_port=9995 --dataset="CIFAR10/clients-5.0" --client_dataset=client19 --num_rounds=250 --num_epochs=2 > client19.out &
sleep 3