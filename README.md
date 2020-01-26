# Deep Learning Exploration (deepx)

### Introduction

This repository contains some work that I have done to port Python based models to C++.
This is completely exploratory and just to maintain my curiosity.  

* Deep Reinforcement Learning with LibTorch.
* Natural Language Processing with LibTorch.
* gRPC based client and server to with OpenAI Gym. This opens up LibTorch to used in many exciting ways.

### Protobuf and gRPC

Setting up C++ versions of protobuf and grpc to compatiable with cmake is indeed exploratory.
The setup that worked for me:

1. Install protobuf from source: https://github.com/protocolbuffers/protobuf/blob/master/src/README.md

2. Install grpc:
    ```bash
    $ git clone -b $(curl -L https://grpc.io/release) https://github.com/grpc/grpc
    $ cd grpc
    $ git submodule update --init
    $ make && sudo make install
    ```

3. Use cmake configurations as described in: https://github.com/alandtsang/cppdemo/tree/master/src/grpcdemo 

4. Setup gRPC Python as described in: https://grpc.io/docs/tutorials/basic/python/

5. gym_env_pb2*.py files are already available in rl-cpp directory. But, if you feel like generating them
(may be some changes in gym_env.proto), use the following command within the rl-cpp directory:

    ```bash
    $ python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. gym_env.proto
    ```
6. To start gRPC Gym server:
    ```bash
    $ python python gym_server.py
    ```
    
7. Download latest LibTorch C++ from https://pytorch.org/get-started/locally/

8. Set CMAKE_PREFIX_PATH to the unzipped location.
    
### DQN - CartPole Example

To execute DQN - CartPole Example

```bash  
$ mkdir build
$ cd build
$ cmake ..
$ make -j2
# <on terminal 1>
$ python python gym_server.py
# <on termina 2>
$ ./deepx_cpp

CUDA available! Training on GPU.
frame_id: 300 mean_value: 23.4
frame_id: 400 mean_value: 19.8
frame_id: 500 mean_value: 21.5
frame_id: 600 mean_value: 28.1
...
frame_id: 9300 mean_value: 221.6
frame_id: 9400 mean_value: 222.2
frame_id: 9500 mean_value: 222.2
frame_id: 9600 mean_value: 222.3
frame_id: 9700 mean_value: 222.3
frame_id: 9800 mean_value: 222.8
frame_id: 9900 mean_value: 222.8
frame_id: 10000 mean_value: 207.2
```
 
### Ubuntu 18.04 fixes
* https://github.com/glassechidna/zxing-cpp/issues/43
* https://github.com/pytorch/pytorch/issues/19353
* https://abseil.io/docs/cpp/guides/flags


Saminda Abeyruwan (samindaa@gmail.com)
