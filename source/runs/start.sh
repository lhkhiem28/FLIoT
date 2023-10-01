cd ../main/

nohup python server.py --server_port=9999 --dataset="CIFAR10/clients-5.0" --num_rounds=500 --num_epochs=1 > server.out &
sleep 30
nohup python client.py --server_port=9999 --dataset="CIFAR10/clients-5.0" --client_dataset=client0 --num_rounds=500 --num_epochs=1 > client0.out &
sleep 3
nohup python client.py --server_port=9999 --dataset="CIFAR10/clients-5.0" --client_dataset=client1 --num_rounds=500 --num_epochs=1 > client1.out &
sleep 3
nohup python client.py --server_port=9999 --dataset="CIFAR10/clients-5.0" --client_dataset=client2 --num_rounds=500 --num_epochs=1 > client2.out &
sleep 3
nohup python client.py --server_port=9999 --dataset="CIFAR10/clients-5.0" --client_dataset=client3 --num_rounds=500 --num_epochs=1 > client3.out &
sleep 3
nohup python client.py --server_port=9999 --dataset="CIFAR10/clients-5.0" --client_dataset=client4 --num_rounds=500 --num_epochs=1 > client4.out &
sleep 3
nohup python client.py --server_port=9999 --dataset="CIFAR10/clients-5.0" --client_dataset=client5 --num_rounds=500 --num_epochs=1 > client5.out &
sleep 3
nohup python client.py --server_port=9999 --dataset="CIFAR10/clients-5.0" --client_dataset=client6 --num_rounds=500 --num_epochs=1 > client6.out &
sleep 3
nohup python client.py --server_port=9999 --dataset="CIFAR10/clients-5.0" --client_dataset=client7 --num_rounds=500 --num_epochs=1 > client7.out &
sleep 3
nohup python client.py --server_port=9999 --dataset="CIFAR10/clients-5.0" --client_dataset=client8 --num_rounds=500 --num_epochs=1 > client8.out &
sleep 3
nohup python client.py --server_port=9999 --dataset="CIFAR10/clients-5.0" --client_dataset=client9 --num_rounds=500 --num_epochs=1 > client9.out &
sleep 3