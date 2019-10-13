cmake_minimum_required(VERSION 3.10.2)
project(deepx_cpp)

set(CMAKE_CXX_STANDARD 14)
#set(CMAKE_PREFIX_PATH "./libtorch;./libabseil")
set(CMAKE_PREFIX_PATH "./libtorch")

#find_package(absl REQUIRED)
find_package(Torch REQUIRED)
find_package (Python3 COMPONENTS Interpreter Development)

file(GLOB MODELS_HEADERS ./vision/torchvision/csrc/models/*.h)
file(GLOB MODELS_SOURCES ./vision/torchvision/csrc/models/*.cpp)

include_directories(${TORCH_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS} ./matplotlib-cpp ./vision/torchvision/csrc/models)
add_subdirectory(abseil-cpp)

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
add_executable(deepx_cpp ${MODELS_HEADERS} ${MODELS_SOURCES} torch-cpp/cifar.cpp torch-cpp/cifar.h torch-cpp/vision_main.cpp)

target_link_libraries(deepx_cpp ${TORCH_LIBRARIES} ${Python3_LIBRARIES} absl::base absl::flags absl::flags_parse absl::strings absl::hash absl::container absl::utility absl::memory absl::meta absl::algorithm)