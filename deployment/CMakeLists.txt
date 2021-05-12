cmake_minimum_required(VERSION 3.0)
set(RE_PATH "$ENV{RE_PATH}")
set(RE_SRC_PATH "${RE_PATH}/src")

# Put Worker libs into RE
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${RE_PATH}/lib")
#add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/cpustatworker")
#add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/memstatworker")
#add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/fftworker")
#add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/particlefilterworker")
#add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/conventionalbfworker")
#add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/caponbfgradworker")
#add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/caponbfbruteworker")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/caponbfsubspaceworker")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/caponbfcholeskyworker")
#add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/marginalisedpfworker")
#add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/swarmoptimisedpfworker")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/adaptiveswarmoptimisedpfworker")

#add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/testworker")
#add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/trackingworker")
#add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/optronicsworker")
#add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/optronicstestworker")
#add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/scenmanworker")


