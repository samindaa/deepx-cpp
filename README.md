# Deep Learning Exploration (deepx)

### Introduction

This repository contains some work that I have done to port Python based models to C++.
This is completely exploratory and just to maintain my curiosity.  

* Deep Reinforcement Learning with LibTorch.
* Natural Language Modeling with LibTorch.
* gRPC based client to communicate with OpenAI Gym.

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


 
### Ubuntu 18.04 fixes
* https://github.com/glassechidna/zxing-cpp/issues/43
* https://github.com/pytorch/pytorch/issues/19353
* https://abseil.io/docs/cpp/guides/flags