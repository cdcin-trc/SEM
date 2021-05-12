cmake_minimum_required (VERSION 2.6)

cmake_host_system_information(RESULT _host_name QUERY HOSTNAME)
#message("-- _host_name variable is set to: " ${_host_name})

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

set(PROJ_NAME adaptiveswarmoptimisedpfworker)
project(${PROJ_NAME})

find_library(RE_CORE_LIBRARIES re_core "${RE_PATH}/lib")
find_library(RE_CPU_WORKER_LIBRARIES cpu_worker "${RE_PATH}/lib")

link_directories( $ENV{LD_LIBRARY_PATH} )


set(Atlas_INCLUDE_SEARCH_PATHS
  /usr/include
  /usr/include/atlas
  /usr/include/atlas-base
  /usr/include/atlas-x86_64-base
  $ENV{Atlas_ROOT_DIR}
  $ENV{Atlas_ROOT_DIR}/include
)

set(Atlas_LIB_SEARCH_PATHS
  /lib64
  /usr/lib/x86_64-linux-gnu
  /usr/lib/atlas
  /usr/lib/atlas-base
  /usr/lib64
  /usr/lib64/atlas
  $ENV{Atlas_ROOT_DIR}
  $ENV{Atlas_ROOT_DIR}/lib
)

find_path(Atlas_CBLAS_INCLUDE_DIR   NAMES cblas.h   PATHS ${Atlas_INCLUDE_SEARCH_PATHS})
find_path(Atlas_CLAPACK_INCLUDE_DIR NAMES clapack.h PATHS ${Atlas_INCLUDE_SEARCH_PATHS})

find_library(Atlas_CBLAS_LIBRARY  NAMES ptcblas_r ptcblas cblas_r cblas blas   PATHS ${Atlas_LIB_SEARCH_PATHS})
find_library(Atlas_BLAS_LIBRARY   NAMES atlas_r atlas tatlas satlas            PATHS ${Atlas_LIB_SEARCH_PATHS})
find_library(Atlas_LAPACK_LIBRARY NAMES alapack_r alapack lapack_atlas lapacke PATHS ${Atlas_LIB_SEARCH_PATHS})

set(LOOKED_FOR
  Atlas_CBLAS_INCLUDE_DIR
  Atlas_CLAPACK_INCLUDE_DIR
  Atlas_CBLAS_LIBRARY
  Atlas_BLAS_LIBRARY
  Atlas_LAPACK_LIBRARY
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Atlas DEFAULT_MSG ${LOOKED_FOR})

if(ATLAS_FOUND)
  set(Atlas_INCLUDE_DIR ${Atlas_CBLAS_INCLUDE_DIR} ${Atlas_CLAPACK_INCLUDE_DIR})
  set(Atlas_LIBRARIES ${Atlas_LAPACK_LIBRARY} ${Atlas_CBLAS_LIBRARY} ${Atlas_BLAS_LIBRARY})
  mark_as_advanced(${LOOKED_FOR})
  message(STATUS "${_host_name} Found Atlas (include: ${Atlas_CBLAS_INCLUDE_DIR}, library: ${Atlas_BLAS_LIBRARY})")
else()
  message(STATUS "${_host_name} Could not find Atlas (include: ${Atlas_CBLAS_INCLUDE_DIR}, library: ${Atlas_BLAS_LIBRARY})")
endif(ATLAS_FOUND)





#find_library(BLAS_LIB blas required)
#find_library(CBLAS_LIB cblas required)
#find_library(ATLAS_LIB atlas required)
#find_library(LAPACK_LIB lapack required)


set(SOURCE
	${CMAKE_CURRENT_SOURCE_DIR}/adaptiveswarmoptimisedpfworker.cpp
)

set(HEADER
	${CMAKE_CURRENT_SOURCE_DIR}/adaptiveswarmoptimisedpfworker.h
	${CMAKE_CURRENT_SOURCE_DIR}/apso_cuda.h	
)

find_package(CUDA)

if(NOT CUDA_FOUND)
	set(SOURCE ${SOURCE}
		${CMAKE_CURRENT_SOURCE_DIR}/apso_cuda.cpp		
	)
endif()

if(CUDA_FOUND)
	message(STATUS "${_host_name} FOUND CUDA")
	add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/cuda")
endif()

add_library(${PROJ_NAME} SHARED ${SOURCE} ${HEADERS})

# Include the source directory for RE
target_include_directories(${PROJ_NAME} PRIVATE ${RE_SRC_PATH})

# Link against the re_core library
target_link_libraries(${PROJ_NAME} PRIVATE ${RE_CORE_LIBRARIES})

# Link against cpu_worker
target_link_libraries(${PROJ_NAME} PRIVATE ${RE_CPU_WORKER_LIBRARIES})

# link against the cblas library
if(ATLAS_FOUND)
	message(STATUS "${_host_name} link against the cblas library:")
	target_link_libraries(${PROJ_NAME} PUBLIC ${Atlas_BLAS_LIBRARY})
	target_link_libraries(${PROJ_NAME} PUBLIC ${Atlas_CBLAS_LIBRARY})
	target_link_libraries(${PROJ_NAME} PUBLIC ${ATLAS_LIB})
	target_link_libraries(${PROJ_NAME} PUBLIC ${Atlas_LAPACK_LIBRARY})
endif(ATLAS_FOUND)

# link against the cuda libraries if found
if(CUDA_FOUND)
	message(STATUS "${_host_name} link against the cuda libraries:")
	target_link_libraries(${PROJ_NAME} PUBLIC apso_cuda_lib)
	target_link_libraries(${PROJ_NAME} PUBLIC ${CUDA_LIBRARIES})	
	target_link_libraries(${PROJ_NAME} PUBLIC ${CUDA_CUBLAS_LIBRARIES})	
	target_link_libraries(${PROJ_NAME} PUBLIC ${CUDA_cusolver_LIBRARY})	
	target_link_libraries(${PROJ_NAME} PUBLIC ${CUDA_curand_LIBRARY})		
endif()







