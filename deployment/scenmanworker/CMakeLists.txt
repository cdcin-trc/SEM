cmake_minimum_required (VERSION 2.6)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)


set(PROJ_NAME scenmanworker)
project(${PROJ_NAME})

find_library(RE_CORE_LIBRARIES re_core "${RE_PATH}/lib")

set(SOURCE
	${CMAKE_CURRENT_SOURCE_DIR}/scenmanworker.cpp
)

set(HEADER
	${CMAKE_CURRENT_SOURCE_DIR}/scenmanworker.h
)

add_library(${PROJ_NAME} SHARED ${SOURCE} ${HEADERS})

# Include the source directory for RE
target_include_directories(${PROJ_NAME} PRIVATE ${RE_SRC_PATH})

# Link against the re_core library
target_link_libraries(${PROJ_NAME} PRIVATE ${RE_CORE_LIBRARIES})


# Link against cpu_worker
target_link_libraries(${PROJ_NAME} PRIVATE ${RE_CPU_WORKER_LIBRARIES})
