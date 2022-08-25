+++

title = "Orchesrating Docker Swarm"
description =  "Dance my docker, dance"
date = "2020-10-01"
author = "Jay Vala"
tags = ["docker", "docker-swarm", "orchestration"]
+++

Initalize a swarm on a node which you can prefer, to create a manager. The architecture of the docker swarm orchestration is that it has a manager node (which is a whole node or computer). Then there are worker nodes which can be controlled or manage by the manager node. 

![Architecture](https://camo.githubusercontent.com/2414d3a74f17d5601aa0abb140ca97ea3eab8ad4/68747470733a2f2f692e706f7374696d672e63632f4d4b4850383959792f737761726d2e6a7067)

(image: https://blog.knoldus.com/orchestration-with-docker-swarm/)


In the diagram above the swam manager is a node which is used to control the swam nodes. In general terminology swarm manager is refered to as `manager` and the swarm nodes are refered to as `workers` or `worker nodes`.

### Create a swarm

To create a swam first we need to create a manager. SSH into a machine and run the command

```bash
docker swarm init --advertise-addr <ip-address>
```

In the above command we tell docker to initalize a docker swarm for us, but when we want to manage the worker nodes or when we want to communicate with the worker nodes we need an `IP` address for the manager to publish in order for workers to connect to our managers, so the command argument `--advertise-addr` will advertise the `IP` address that follows the command for workder nodes of other managers to connect to this manager node.

> In case we want to use only one node, we can simply run `docker swarm init`

```bash
$ docker swarm init --advertise-addr 192.168.0.23 
Swarm initialized: current node (v6l24ilbz1bq72ad8tklhipxx) is now a manager.

To add a worker to this swarm, run the following command:

    docker swarm join --token SWMTKN-1-04i6f00j63iyiqts65tynfapes4qx6bhxap21lod0k69jzmol3-5tjti3fxaqg94qkv3zf86k6ai 192.168.0.23:2377

To add a manager to this swarm, run 'docker swarm join-token manager' and follow the instructions.
```

Now we can use `docker info` to display the information about the current state or our swarm.

```bash
docker info

Client:
 Debug Mode: false
 Plugins:
  app: Docker App (Docker Inc., v0.9.1-beta3)

Server:
 Containers: 0
  Running: 0
  Paused: 0
  Stopped: 0
 Images: 1
 Server Version: 19.03.11
 Storage Driver: overlay2
  Backing Filesystem: xfs
  Supports d_type: true
  Native Overlay Diff: true
 Logging Driver: json-file
 Cgroup Driver: cgroupfs
 Plugins:
  Volume: local
  Network: bridge host ipvlan macvlan null overlay
  Log: awslogs fluentd gcplogs gelf journald json-file local logentries splunk syslog
 Swarm: active
  NodeID: v6l24ilbz1bq72ad8tklhipxx
  Is Manager: true
  ClusterID: 6a4b0em1ndyg52odc43guyr1i
  Managers: 1
  Nodes: 2
  Default Address Pool: 10.0.0.0/8  
  SubnetSize: 24
  Data Path Port: 4789
  Orchestration:
   Task History Retention Limit: 5
  Raft:
   Snapshot Interval: 10000
   Number of Old Snapshots to Retain: 0
   Heartbeat Tick: 1
   Election Tick: 10
  Dispatcher:
   Heartbeat Period: 5 seconds
  CA Configuration:
   Expiry Duration: 3 months
   Force Rotate: 0
  Autolock Managers: false
  Root Rotation In Progress: false
  Node Address: 192.168.0.23
  Manager Addresses:
   192.168.0.23:2377
 Runtimes: runc
 Default Runtime: runc
 Init Binary: docker-init
 containerd version: 7ad184331fa3e55e52b890ea95e65ba581ae3429
 runc version: dc9208a3303feef5b3839f4323d9beb36df0a9dd
 init version: fec3683
 Security Options:
  apparmor
  seccomp
   Profile: default
 Kernel Version: 4.4.0-186-generic
 Operating System: Alpine Linux v3.12 (containerized)
 OSType: linux
 Architecture: x86_64
 CPUs: 8
 Total Memory: 31.4GiB
 Name: node1
 ID: XHBK:ACSE:UHWV:LFEY:MCE2:BAOA:775R:DK6P:CBYL:3RDJ:2WHV:QQ4M
 Docker Root Dir: /var/lib/docker
 Debug Mode: true
  File Descriptors: 38
  Goroutines: 164
  System Time: 2020-09-01T08:25:22.272726765Z
  EventsListeners: 0
 Registry: https://index.docker.io/v1/
 Labels:
 Experimental: true
 Insecure Registries:
  127.0.0.1
  127.0.0.0/8
 Live Restore Enabled: false
 Product License: Community Engine

WARNING: API is accessible on http://0.0.0.0:2375 without encryption.
         Access to the remote API is equivalent to root access on the host. Refer
         to the 'Docker daemon attack surface' section in the documentation for
         more information: https://docs.docker.com/engine/security/security/#docker-daemon-attack-surface
WARNING: No swap limit support
WARNING: bridge-nf-call-iptables is disabled
WARNING: bridge-nf-call-ip6tables is disabled
```

we can also run `docker node ls` to see information about all the nodes.

```bash
docker node ls

ID                            HOSTNAME            STATUS              AVAILABILITY        MANAGER STATUS      ENGINE VERSION
v6l24ilbz1bq72ad8tklhipxx *   node1               Ready               Active              Leader              19.03.11
pw51r1us9l1ou62pdh3ff3dp2     node2               Ready               Active                                  19.03.11
```

> The * indicates we are connected to the that node.


### Add nodes to the swarm

To add nodes to the swarm run the command displayed when we initalized the swarm.

```bash
docker swarm join --token SWMTKN-1-04i6f00j63iyiqts65tynfapes4qx6bhxap21lod0k69jzmol3-5tjti3fxaqg94qkv3zf86k6ai 192.168.0.23:2377
```

This will spawn up a node (worker). Please note that this command has to be run on a different physical computer. Also the `IP` address specified `192.168.0.23:2377` should be accessible to this machine where we are spawning a worker node.


If the command from the init swarm command is not accessible you can run `docker swarm join-token worker` to get the command.

Once the node is setup. We can use `docker node ls` to see how many nodes we have. I had already setup a node so you can see the output in my previous `docker node ls` command.


### Deploy a service

I will run a simple `helloworld` program which will ping `google.com`

To create a service we use `docker service create` command

```bash
docker create servec --replica 3 --name helloworld alpine ping jayvala.in
```

The above command will create a service named `helloworld` with base image as `alpine` and the command as the last argument `ping google.com`

The `replicas` flag is important here. It specifices on how many instances we want to run this services. Its good practice to have more than one replica as it helps in high availibility. 

> For the swarm manager too can be an instance so once you execute this command it can spawn the service on manager too.

We can use `docker service ls` command to see the list of services running.

To inspect the service running on our swarm we can use 

```bash
$ docker service inspect --pretty helloworld

ID:             cnpv2ykky1lmm5vpdd9id0u0q
Name:           helloworld
Service Mode:   Replicated
 Replicas:      3
Placement:
UpdateConfig:
 Parallelism:   1
 On failure:    pause
 Monitoring Period: 5s
 Max failure ratio: 0
 Update order:      stop-first
RollbackConfig:
 Parallelism:   1
 On failure:    pause
 Monitoring Period: 5s
 Max failure ratio: 0
 Rollback order:    stop-first
ContainerSpec:
 Image:         alpine:latest@sha256:185518070891758909c9f839cf4ca393ee977ac378609f700f60a771a2dfe321
 Args:          ping jayvala.com 
 Init:          false
Resources:
Endpoint Mode:  vip
```

Now lets see which nodes in our swarm is running our helloworld service

```bash
$ docker service ps helloworld
ID                  NAME                IMAGE               NODE                DESIRED STATE       CURRENT STATE           ERROR               PORTS
zl1z36sfzr50        helloworld.1        alpine:latest       node2               Running             Running 4 minutes ago                       
b3pb76uuvpsi        helloworld.2        alpine:latest       node1               Running             Running 4 minutes ago                       
21y3swo9n8cj        helloworld.3        alpine:latest       node2               Running             Running 4 minutes ago                       
```
As we can see that we have 3 helloworld services running, one on `node 1` and two on `node 2`. 


Now as the users consuming this services increase we would want to scale up these services.

### Scale up the service

To scale up the we can use the following command

```bash
docker service scale <service name>=<number of tasks>
```

We can se here that we can replicate the tasks (a task is nothing but a single instance of a service). 

```bash
docker service scale helloworld=10

helloworld scaled to 10
overall progress: 10 out of 10 tasks 
1/10: running   [==================================================>] 
2/10: running   [==================================================>] 
3/10: running   [==================================================>] 
4/10: running   [==================================================>] 
5/10: running   [==================================================>] 
6/10: running   [==================================================>] 
7/10: running   [==================================================>] 
8/10: running   [==================================================>] 
9/10: running   [==================================================>] 
10/10: running   [==================================================>] 
verify: Service converged 
```

The above command started 10 more replicas of the helloworld services.

Now to see which nodes are running how many process we can simply use

```bash
docker service ps helloworld

$ docker service ps helloworld
ID                  NAME                IMAGE               NODE                DESIRED STATE       CURRENT STATE                ERROR               PORTS
zl1z36sfzr50        helloworld.1        alpine:latest       node2               Running             Running 11 minutes ago                           
b3pb76uuvpsi        helloworld.2        alpine:latest       node1               Running             Running 11 minutes ago                           
21y3swo9n8cj        helloworld.3        alpine:latest       node2               Running             Running 11 minutes ago                           
5e650ivizaet        helloworld.4        alpine:latest       node2               Running             Running about a minute ago                       
2u0cz08jp9sk        helloworld.5        alpine:latest       node1               Running             Running about a minute ago                       
011vnxwzyjd3        helloworld.6        alpine:latest       node2               Running             Running about a minute ago                       
drwrg7ixa42p        helloworld.7        alpine:latest       node1               Running             Running about a minute ago                       
mcl513va9pes        helloworld.8        alpine:latest       node1               Running             Running about a minute ago                       
6c7lwv522f4x        helloworld.9        alpine:latest       node2               Running             Running about a minute ago                       
4441rdl391rv        helloworld.10       alpine:latest       node1               Running             Running about a minute ago      
```

Now on manager node we can do `docker ps` and it will list out all the tasks its been running, we can do the same on any node.

### Deleting the service

To remove the service completely we can use

```bash
docker service rm helloworld
```

Now we can run the command `docker ps` to verify all our services are deleted or not. 

```bash
docker ps
CONTAINER ID        IMAGE               COMMAND             CREATED             STATUS              PORTS               NAMES
```
