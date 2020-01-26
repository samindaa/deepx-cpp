cmake_minimum_required(VERSION 3.10.2)
project(deepx_cpp)

set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_PREFIX_PATH "./libtorch;./libabseil")
set(CMAKE_PREFIX_PATH "./libtorch")

# Find Protobuf
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
find_package(Protobuf REQUIRED)
set(_PROTOBUF_LIBPROTOBUF protobuf::libprotobuf)
message(STATUS "Using protobuf ${protobuf_VERSION}")

# Find gRPC
find_package(GRPC REQUIRED)
set(_GRPC_GRPCPP_REFLECTION gRPC::grpc++_reflection)
message(STATUS "Using gRPC ${gRPC_VERSION}")

#find_package(Threads REQUIRED)
#find_package(absl REQUIRED)
find_package(Torch REQUIRED)
#find_package(Python3 COMPONENTS Interpreter Development)


# Proto file
message(STATUS "CMAKE_CURRENT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}")
set(gym_env ${CMAKE_CURRENT_SOURCE_DIR}/rl-cpp/gym_env.proto)

set(PROTO_SRC_DIR ${CMAKE_CURRENT_BINARY_DIR}/proto-src)
file(MAKE_DIRECTORY ${PROTO_SRC_DIR})

# Include generated *.pb.h files
include_directories(${PROTO_SRC_DIR})

# Generated sources
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_SRC_DIR} ${gym_env})
grpc_generate_cpp(GRPC_SRCS GRPC_HDRS ${PROTO_SRC_DIR} ${gym_env})


#file(GLOB MODELS_HEADERS ./vision/torchvision/csrc/models/*.h)
#file(GLOB MODELS_SOURCES ./vision/torchvision/csrc/models/*.cpp)

#include_directories(${TORCH_INCLUDE_DIRS}
#        ${Python3_INCLUDE_DIRS}
#        ./matplotlib-cpp
#        ./vision/torchvision/csrc/models
#        ${CMAKE_CURRENT_BINARY_DIR})
#add_subdirectory(abseil-cpp)

#add_executable(deepx_cpp aima-cpp/main.cpp)
#add_executable(deepx_cpp torch-cpp/basic_main.cpp)
#add_executable(deepx_cpp torch-cpp/resnet_main.cpp torch-cpp/cifar.h torch-cpp/cifar.cpp)
#add_executable(deepx_cpp torch-cpp/cifar.cpp torch-cpp/cifar.h torch-cpp/mobilenet_main.cpp)
#add_executable(deepx_cpp torch-cpp/cifar.cpp torch-cpp/cifar.h torch-cpp/mobilenetv2_main.cpp)
#add_executable(deepx_cpp torch-cpp/char_rnn_classification_main.cpp)
#add_executable(deepx_cpp torch-cpp/char_rnn_generation_main.cpp)
#add_executable(deepx_cpp torch-cpp/seq2seq_translation_main.cpp)
#add_executable(deepx_cpp torch-cpp/chatbot_main.cpp)
#add_executable(deepx_cpp misc-cpp/matplotlib_main.cpp)
#add_executable(deepx_cpp ${MODELS_HEADERS} ${MODELS_SOURCES} torch-cpp/cifar.cpp torch-cpp/cifar.h torch-cpp/vision_main.cpp ptan-cpp/actions.h)
#add_executable(deepx_cpp ptan-cpp/actions.h ptan-cpp/ptan_main.cpp ptan-cpp/agent.h)
add_executable(deepx_cpp
    rl-cpp/rl_main.cpp
	rl-cpp/replay_buffer.h
	rl-cpp/replay_buffer.cpp
	rl-cpp/dqn.h
	rl-cpp/gym.h
	rl-cpp/gym.cpp
        ${PROTO_SRCS}
        ${GRPC_SRCS} rl-cpp/common.h rl-cpp/common.cpp rl-cpp/dqn.cpp)

target_link_libraries(deepx_cpp
        ${_GRPC_GRPCPP_REFLECTION}
        ${_PROTOBUF_LIBPROTOBUF}
	${TORCH_LIBRARIES}
	#${Python3_LIBRARIES}
	#       absl::base
	# absl::flags
	# absl::flags_parse
	# absl::strings
	#absl::hash
	#absl::container
	#absl::utility
	#absl::memory
	#absl::meta
	#absl::algorithm
        )
