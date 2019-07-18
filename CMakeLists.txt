cmake_minimum_required(VERSION 3.10.2)
project(saminda_aima_cpp)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_PREFIX_PATH "./libtorch;./libabseil")

find_package(absl REQUIRED)
find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})
#add_subdirectory(abseil-cpp)

#add_executable(saminda_aima_cpp aima-cpp/main.cpp)
#add_executable(saminda_aima_cpp torch-cpp/main.cpp)
add_executable(saminda_aima_cpp torch-cpp/main2.cpp)
target_link_libraries(saminda_aima_cpp ${TORCH_LIBRARIES} absl::base absl::flags absl::flags_parse absl::strings absl::hash absl::container absl::utility absl::memory absl::meta absl::algorithm)