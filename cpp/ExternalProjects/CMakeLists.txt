cmake_minimum_required(VERSION 3.5.2)

project(ExternalProjects)

# This will build a subset of dependencies often found at operating-system level
# package managers (e.g. Debian packages managed by aptitude). This file copies
# much of the CMake external dependencies written by ajakhotia in his robotFarm
# repository found here: https://github.com/ajakhotia/robotFarm

include(ExternalProject)

externalproject_add(OpenCVExternalProject
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/opencv
    URL "https://github.com/opencv/opencv/archive/4.1.1.zip"
    DOWNLOAD_NO_PROGRESS ON
    CMAKE_ARGS
    -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
    -DCMAKE_PREFIX_PATH:PATH=${CMAKE_INSTALL_PREFIX}
    -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_INSTALL_PREFIX}
    -DWITH_VTK:BOOL=ON
    -DWITH_NVCUVID:BOOL=OFF)


externalproject_add(Eigen3ExternalProject
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/eigen3
    URL "http://bitbucket.org/eigen/eigen/get/3.3.7.zip"
    DOWNLOAD_NO_PROGRESS ON
    CMAKE_ARGS
    -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
    -DCMAKE_PREFIX_PATH:PATH=${CMAKE_INSTALL_PREFIX}
    -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_INSTALL_PREFIX})


externalproject_add(GoogleTestExternalProject
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/googleTest
    URL "https://github.com/google/googletest/archive/release-1.8.1.zip"
    DOWNLOAD_NO_PROGRESS ON
    CMAKE_ARGS
    -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
    -DCMAKE_PREFIX_PATH:PATH=${CMAKE_INSTALL_PREFIX}
    -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_INSTALL_PREFIX})


externalproject_add(GlogExternalProject
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/glog
    URL "https://github.com/google/glog/archive/v0.4.0.zip"
    DOWNLOAD_NO_PROGRESS ON
    CMAKE_ARGS
    -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
    -DCMAKE_PREFIX_PATH:PATH=${CMAKE_INSTALL_PREFIX}
    -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_INSTALL_PREFIX}
    -DBUILD_SHARED_LIBS:BOOL=ON)


externalproject_add(CeresSolverExternalProject
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/ceresSolver
    URL "https://github.com/ceres-solver/ceres-solver/archive/1.14.0.tar.gz"
    DOWNLOAD_NO_PROGRESS ON
    CMAKE_ARGS
    -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
    -DCMAKE_PREFIX_PATH:PATH=${CMAKE_INSTALL_PREFIX}
    -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_INSTALL_PREFIX})

add_dependencies(CeresSolverExternalProject GlogExternalProject Eigen3ExternalProject)
