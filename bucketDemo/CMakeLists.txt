cmake_minimum_required(VERSION 3.10)
project(BucketSearch)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")

# Search executable
add_executable(search search.cpp)

# OpenMP for parallel processing
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(search PUBLIC OpenMP::OpenMP_CXX)
endif()

message(STATUS "BucketSearch project configured")
