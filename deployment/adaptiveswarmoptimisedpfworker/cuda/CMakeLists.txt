cmake_minimum_required(VERSION 3.6)

set(PROJ_NAME apso_cuda_lib)

project(${PROJ_NAME})


find_package(CUDA REQUIRED)

#set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
#set(CMAKE_CUDA_COMPILER_ENV_VAR "/usr/local/cuda/bin/nvcc")

cuda_add_library(${PROJ_NAME}
	apso_cuda.cu
	SHARED
)

set_target_properties(${PROJ_NAME} PROPERTIES CUDA_SEPERABLE_COMPILATION ON)
#set_target_properties(${PROJ_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
