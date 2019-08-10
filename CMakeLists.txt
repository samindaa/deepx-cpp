cmake_minimum_required(VERSION 3.10.2)
project(deepx_cpp)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_PREFIX_PATH "./libtorch;./libabseil")

find_package(absl REQUIRED)
find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})
#add_subdirectory(abseil-cpp)

#add_executable(deepx_cpp aima-cpp/main.cpp)
#add_executable(deepx_cpp torch-cpp/basic_main.cpp)
#add_executable(deepx_cpp torch-cpp/resnet_main.cpp torch-cpp/cifar.h torch-cpp/cifar.cpp)
#add_executable(deepx_cpp torch-cpp/cifar.cpp torch-cpp/cifar.h torch-cpp/mobilenet_main.cpp)
#add_executable(deepx_cpp torch-cpp/cifar.cpp torch-cpp/cifar.h torch-cpp/mobilenetv2_main.cpp)
#add_executable(deepx_cpp torch-cpp/char_rnn_classification_main.cpp)
add_executable(deepx_cpp torch-cpp/char_rnn_generation_main.cpp)

target_link_libraries(deepx_cpp ${TORCH_LIBRARIES} absl::base absl::flags absl::flags_parse absl::strings absl::hash absl::container absl::utility absl::memory absl::meta absl::algorithm)