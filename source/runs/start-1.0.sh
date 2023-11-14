cd ../main/

nohup python server.py --server_port=9991 --dataset="CIFAR10/clients-1.0" --num_rounds=250 --num_epochs=2 > server-1.0.out &
sleep 30

nohup python client.py --server_port=9991 --dataset="CIFAR10/clients-1.0" --client_dataset=client00 --num_rounds=250 --num_epochs=2 > client00-1.0.out &
sleep 3
nohup python client.py --server_port=9991 --dataset="CIFAR10/clients-1.0" --client_dataset=client01 --num_rounds=250 --num_epochs=2 > client01-1.0.out &
sleep 3
nohup python client.py --server_port=9991 --dataset="CIFAR10/clients-1.0" --client_dataset=client02 --num_rounds=250 --num_epochs=2 > client02-1.0.out &
sleep 3
nohup python client.py --server_port=9991 --dataset="CIFAR10/clients-1.0" --client_dataset=client03 --num_rounds=250 --num_epochs=2 > client03-1.0.out &
sleep 3
nohup python client.py --server_port=9991 --dataset="CIFAR10/clients-1.0" --client_dataset=client04 --num_rounds=250 --num_epochs=2 > client04-1.0.out &
sleep 3
nohup python client.py --server_port=9991 --dataset="CIFAR10/clients-1.0" --client_dataset=client05 --num_rounds=250 --num_epochs=2 > client05-1.0.out &
sleep 3
nohup python client.py --server_port=9991 --dataset="CIFAR10/clients-1.0" --client_dataset=client06 --num_rounds=250 --num_epochs=2 > client06-1.0.out &
sleep 3
nohup python client.py --server_port=9991 --dataset="CIFAR10/clients-1.0" --client_dataset=client07 --num_rounds=250 --num_epochs=2 > client07-1.0.out &
sleep 3
nohup python client.py --server_port=9991 --dataset="CIFAR10/clients-1.0" --client_dataset=client08 --num_rounds=250 --num_epochs=2 > client08-1.0.out &
sleep 3
nohup python client.py --server_port=9991 --dataset="CIFAR10/clients-1.0" --client_dataset=client09 --num_rounds=250 --num_epochs=2 > client09-1.0.out &
sleep 3
nohup python client.py --server_port=9991 --dataset="CIFAR10/clients-1.0" --client_dataset=client10 --num_rounds=250 --num_epochs=2 > client10-1.0.out &
sleep 3
nohup python client.py --server_port=9991 --dataset="CIFAR10/clients-1.0" --client_dataset=client11 --num_rounds=250 --num_epochs=2 > client11-1.0.out &
sleep 3
nohup python client.py --server_port=9991 --dataset="CIFAR10/clients-1.0" --client_dataset=client12 --num_rounds=250 --num_epochs=2 > client12-1.0.out &
sleep 3
nohup python client.py --server_port=9991 --dataset="CIFAR10/clients-1.0" --client_dataset=client13 --num_rounds=250 --num_epochs=2 > client13-1.0.out &
sleep 3
nohup python client.py --server_port=9991 --dataset="CIFAR10/clients-1.0" --client_dataset=client14 --num_rounds=250 --num_epochs=2 > client14-1.0.out &
sleep 3
nohup python client.py --server_port=9991 --dataset="CIFAR10/clients-1.0" --client_dataset=client15 --num_rounds=250 --num_epochs=2 > client15-1.0.out &
sleep 3
nohup python client.py --server_port=9991 --dataset="CIFAR10/clients-1.0" --client_dataset=client16 --num_rounds=250 --num_epochs=2 > client16-1.0.out &
sleep 3
nohup python client.py --server_port=9991 --dataset="CIFAR10/clients-1.0" --client_dataset=client17 --num_rounds=250 --num_epochs=2 > client17-1.0.out &
sleep 3
nohup python client.py --server_port=9991 --dataset="CIFAR10/clients-1.0" --client_dataset=client18 --num_rounds=250 --num_epochs=2 > client18-1.0.out &
sleep 3
nohup python client.py --server_port=9991 --dataset="CIFAR10/clients-1.0" --client_dataset=client19 --num_rounds=250 --num_epochs=2 > client19-1.0.out &
sleep 3