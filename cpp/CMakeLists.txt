# cmake 3.8+ required for C++17
cmake_minimum_required(VERSION 3.8.2)

project(Vision)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

add_subdirectory(CameraCalibration)
add_subdirectory(Geometry)
add_subdirectory(Image)
add_subdirectory(Panorama)
add_subdirectory(Samples)
add_subdirectory(Solvers)
