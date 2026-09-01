cmake_minimum_required(VERSION 3.22 FATAL_ERROR)

# 编译器探测 (优先环境变量，次选系统路径，最后回退到默认 conda 路径)
if(NOT DEFINED CMAKE_CUDA_COMPILER)
    find_program(SYSTEM_NVCC nvcc PATHS /usr/local/cuda/bin /home/lanlu/miniconda3/envs/rapids_raft/bin)
    if(SYSTEM_NVCC)
        set(CMAKE_CUDA_COMPILER ${SYSTEM_NVCC})
    endif()
endif()
if(NOT DEFINED CMAKE_CXX_COMPILER)
    find_program(SYSTEM_CXX g++ x86_64-conda-linux-gnu-g++ PATHS /usr/bin /home/lanlu/miniconda3/envs/rapids_raft/bin)
    if(SYSTEM_CXX)
        set(CMAKE_CXX_COMPILER ${SYSTEM_CXX})
    endif()
endif()
if(NOT DEFINED CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

project(
    buildBucket
    LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
# 自动探测或支持常用架构 (T4: 75, V100: 70, A100: 80, RTX30/A10: 86)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 70 80 86)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma -msse2 -ftree-vectorize -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free -fopenmp -fopenmp-simd -funroll-loops -Wfatal-errors -DUSE_AVX2 -DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE -I/content/raft/cpp/include -I/content/rmm/include -I/content/cuco/include -I/content/cutlass/include")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE --extended-lambda --expt-relaxed-constexpr -I/content/raft/cpp/include -I/content/rmm/include -I/content/cuco/include -I/content/cutlass/include")

find_package(Boost QUIET COMPONENTS program_options)
if(NOT Boost_FOUND)
    if(DEFINED ENV{CONDA_PREFIX})
        set(Boost_INCLUDE_DIRS $ENV{CONDA_PREFIX}/include)
        set(Boost_LIBRARY_DIRS $ENV{CONDA_PREFIX}/lib)
    endif()
endif()

find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

# RAFT 查找 (支持 config package 或直接 include headers)
find_package(raft CONFIG QUIET
    PATHS 
        $ENV{CONDA_PREFIX}
        $ENV{RAFT_PATH}
        /home/lanlu/raft/cpp/build
        /content/raft/cpp/build
        /usr/local
)

if(NOT TARGET raft::raft)
    add_library(raft_interface INTERFACE)
    add_library(raft::raft ALIAS raft_interface)
endif()

if(DEFINED ENV{CONDA_PREFIX})
    set(CONDA_INCLUDE $ENV{CONDA_PREFIX}/include)
    set(CONDA_LIB $ENV{CONDA_PREFIX}/lib)
else()
    set(CONDA_INCLUDE /usr/local/include)
    set(CONDA_LIB /usr/local/lib)
endif()

include_directories(
    SYSTEM
    ${CONDA_INCLUDE}
    /content/raft/cpp/include
    /content/rmm/include
    /content/cuco/include
    /content/cutlass/include
    /content/raft/cpp/build/_deps/nlohmann_json-src/include
    /home/lanlu/raft/cpp/include/
    /home/lanlu/miniconda3/envs/rapids_raft/targets/x86_64-linux/include
    /home/lanlu/miniconda3/envs/rapids_raft/include
    /home/lanlu/miniconda3/envs/rapids_raft/include/rapids
    /home/lanlu/miniconda3/envs/rapids_raft/include/rapids/libcudacxx
    /home/lanlu/raft/cpp/build/_deps/nlohmann_json-src/include
    ${Boost_INCLUDE_DIRS}
)

link_directories(
    /home/lanlu/raft/cpp/build/_deps/benchmark-build/src
    ${Boost_LIBRARY_DIRS}
    ${CONDA_LIB}
)

set(BUILD_SRC_PATH ${CMAKE_CURRENT_SOURCE_DIR}/src)

set(BUILD_SOURCES
    bucket_build.cu)
list(TRANSFORM BUILD_SOURCES PREPEND ${BUILD_SRC_PATH}/)

add_executable(bucket ${BUILD_SOURCES})

target_link_libraries(bucket
    PRIVATE
    CUDA::cudart
    CUDA::cublas
    OpenMP::OpenMP_CXX
    raft::raft
    fmt
    boost_program_options
)

set_target_properties(bucket PROPERTIES
    BUILD_RPATH ${CONDA_LIB}
)

# bucket2: bucket.cu (with tensor core matmul + per-vector KNN)
add_executable(bucket2 ${BUILD_SRC_PATH}/bucket.cu)

target_link_libraries(bucket2
    PRIVATE
    CUDA::cudart
    CUDA::cublas
    OpenMP::OpenMP_CXX
    raft::raft
    fmt
    boost_program_options
)

set_target_properties(bucket2 PROPERTIES
    BUILD_RPATH ${CONDA_LIB}
)

# bucket_reordered: bucket_reordered.cu (with cache-optimized contiguous bucket slice layout)
add_executable(bucket_reordered ${BUILD_SRC_PATH}/bucket_reordered.cu)

target_link_libraries(bucket_reordered
    PRIVATE
    CUDA::cudart
    CUDA::cublas
    OpenMP::OpenMP_CXX
    raft::raft
    fmt
    boost_program_options
)

set_target_properties(bucket_reordered PROPERTIES
    BUILD_RPATH ${CONDA_LIB}
)

# optimize: standalone tool that runs CAGRA sort_knn_graph + optimize on
# the vector_knn.bin produced by bucket.cu, yielding a pruned CAGRA graph.
add_executable(optimize ${BUILD_SRC_PATH}/optimize.cu)

target_link_libraries(optimize
    PRIVATE
    CUDA::cudart
    OpenMP::OpenMP_CXX
    raft::raft
    fmt
    boost_program_options
)

set_target_properties(optimize PROPERTIES
    BUILD_RPATH ${CONDA_LIB}
)

# optimize_chunked: chunked reverse-graph + merge for huge graphs that don't
# fit on a single GPU. Implements 3 methods (A: kernel-filter, B: pre-shuffle,
# C: bucket-aligned A) selectable via --method.
add_executable(optimize_chunked ${BUILD_SRC_PATH}/optimize_chunked.cu)

target_link_libraries(optimize_chunked
    PRIVATE
    CUDA::cudart
    OpenMP::OpenMP_CXX
    fmt
    boost_program_options
)

set_target_properties(optimize_chunked PROPERTIES
    BUILD_RPATH ${CONDA_LIB}
)

# reorder: CPU-only tool that reorders dataset + vector_knn.bin by bucket
# layout (consecutive new IDs per bucket), enabling optimize_chunked -m C.
add_executable(reorder ${BUILD_SRC_PATH}/reorder.cpp)

target_link_libraries(reorder
    PRIVATE
    OpenMP::OpenMP_CXX
    boost_program_options
)

set_target_properties(reorder PROPERTIES
    BUILD_RPATH ${CONDA_LIB}
)
