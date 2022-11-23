# Docker Commands

## Software Install

```bash
    source .bashrc

    # create a new environment
    conda create --name kafka --clone lewisu

    # upgrade pip
    pip install --upgrade pip

    # Install packages (Week2)
    yum -y update
    yum install -y package-name

    # Install needed development tools
    yum -y update
    yum install -y vim nano epel-release net-tools iproute
    yum groupinstall -y "Development Tools"
    yum install -y python3

    python3 -m pip install --upgrade pip
    pip install --user numpy pandas

    # Save new image containing changes
    docker commit centos centos-jeff

    # Pull the python container image
    docker pull python

    pip install dnspython pymongo
    pip install kafka-python  # centos, airline, hotel, car

    conda install -c conda-forge kafka-python
```

## Commands

```bash
    docker network create --driver bridge centos-net

    # Run container instance in background
    # -i means interactive mode
    # -t gives you a terminal
    # -d (daemon mode) keeps the container running in the background
    docker run -it -d --name centos --network centos-net centos-jeff bash

    # Connect to container
    docker exec -it centos /bin/bash
    docker attach centos

    docker network inspect centos-net
    docker network inspect bridge

    # Connect to container
    docker exec -it pbs /bin/bash
    docker attach pbs

    # Container status
    docker container ls
    docker ps
    docker ps -a --filter ancestor=centos
    docker-compose ps

    # Not using LoadBalancer
    docker start logger centos airline hotel car
    docker stop logger centos airline hotel car

    # Stop the Docker containers for Confluent
    docker container stop $(docker container ls -a -q -f "label=io.confluent.docker")

    docker ps -a --filter ancestor=centos
```

## More Commands

```bash
    # Copy files
    cd /root
    cp -r week6 week7

    # Copy files from Mac to Linux
    scp *.py jeff@morpheus:/home/jeff
    scp -r week7 jeff@morpheus:/home/jeff

    # Copy files from Linux to Mac
    scp -r jeff@morpheus:/home/jeff/Code/week4 .

    # Copy files from host to docker container
    docker cp week5 centos:/root

    # Copy files from container to linux host
    docker cp centos:/root/week4 .

    # Find ip address of the containers
    ip a
    docker inspect

    # Find mac address
    cat /sys/class/net/wlp1s0/address
    cat /sys/class/net/eth0/address

    hostname
    hostname name

    # NetworkManager Tools
    nmcli

    head HelloWorldServer.py
    tail HelloWorldServer.py
```
