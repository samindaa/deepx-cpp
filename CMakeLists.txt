cmake_minimum_required(VERSION 3.14)
project(saminda_aima_cpp)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_PREFIX_PATH ./libtorch/torch)

add_subdirectory(abseil-cpp)
find_package(Torch REQUIRED)

#add_executable(saminda_aima_cpp aima-cpp/main.cpp)
add_executable(saminda_aima_cpp torch-cpp/main.cpp)
target_link_libraries(saminda_aima_cpp
        absl::base
        absl::flags
        absl::strings
        absl::hash
        absl::container
        absl::utility
        absl::memory
        absl::meta
        absl::algorithm
        "${TORCH_LIBRARIES}")