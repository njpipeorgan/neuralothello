cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(othello)

find_package(Torch REQUIRED)

add_executable(othello src/main.cpp)
target_link_libraries(othello "${TORCH_LIBRARIES}")
set_property(TARGET othello PROPERTY CXX_STANDARD 17)

SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fopenmp -O3 -march=native")
