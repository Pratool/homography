cmake_minimum_required(VERSION 3.6)

project(ExternalProjects)

# This will build a subset of dependencies often found at operating-system level
# package managers (e.g. Debian packages managed by aptitude). This file copies
# much of the CMake external dependencies written by ajakhotia in his robotFarm
# repository found here: https://github.com/ajakhotia/robotFarm

include(ExternalProject)

externalproject_add(OpenCVExternalProject
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/opencv
    URL "https://github.com/opencv/opencv/archive/4.1.1.zip"
    URL_HASH SHA256=d34985c7b4283519c032e3585b30846644f56c81acba35693295ff8930f080aa
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
    URL_HASH SHA256=d34985c7b4283519c032e3585b30846644f56c81acba35693295ff8930f080aa
    DOWNLOAD_NO_PROGRESS ON
    CMAKE_ARGS
        -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
        -DCMAKE_PREFIX_PATH:PATH=${CMAKE_INSTALL_PREFIX}
        -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_INSTALL_PREFIX})


externalproject_add(GoogleTestExternalProject
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/googleTest
    URL "https://github.com/google/googletest/archive/release-1.8.1.zip"
    URL_HASH SHA256=927827c183d01734cc5cfef85e0ff3f5a92ffe6188e0d18e909c5efebf28a0c7
    DOWNLOAD_NO_PROGRESS ON
    CMAKE_ARGS
        -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
        -DCMAKE_PREFIX_PATH:PATH=${CMAKE_INSTALL_PREFIX}
        -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_INSTALL_PREFIX})


externalproject_add(GlogExternalProject
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/glog
    URL "https://github.com/google/glog/archive/v0.4.0.zip"
    URL_HASH SHA256=9e1b54eb2782f53cd8af107ecf08d2ab64b8d0dc2b7f5594472f3bd63ca85cdc
    DOWNLOAD_NO_PROGRESS ON
    CMAKE_ARGS
        -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
        -DCMAKE_PREFIX_PATH:PATH=${CMAKE_INSTALL_PREFIX}
        -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_INSTALL_PREFIX}
        -DBUILD_SHARED_LIBS:BOOL=ON)


externalproject_add(CeresSolverExternalProject
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/ceresSolver
    URL "https://github.com/ceres-solver/ceres-solver/archive/1.14.0.tar.gz"
    URL_HASH SHA256=1296330fcf1e09e6c2f926301916f64d4a4c5c0ff12d460a9bc5d4c48411518f
    DOWNLOAD_NO_PROGRESS ON
    CMAKE_ARGS
        -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
        -DCMAKE_PREFIX_PATH:PATH=${CMAKE_INSTALL_PREFIX}
        -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_INSTALL_PREFIX})

add_dependencies(CeresSolverExternalProject GlogExternalProject Eigen3ExternalProject)


externalproject_add(BoostExternalProject
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/boost
    URL "https://dl.bintray.com/boostorg/release/1.72.0/source/boost_1_72_0.tar.bz2"
    URL_HASH SHA256=59c9b274bc451cf91a9ba1dd2c7fdcaf5d60b1b3aa83f2c9fa143417cc660722
    DOWNLOAD_NO_PROGRESS ON
    BUILD_IN_SOURCE ON
    CONFIGURE_COMMAND
        LD_LIBRARY_PATH=${CMAKE_INSTALL_PREFIX}/lib:$ENV{LD_LIBRARY_PATH}
        PATH=${CMAKE_INSTALL_PREFIX}/bin:$ENV{PATH}
        ./bootstrap.sh
        --prefix=${CMAKE_INSTALL_PREFIX}
        --with-python-root=${CMAKE_INSTALL_PREFIX}
        --with-python=python3
    BUILD_COMMAND ./b2 install
        --prefix=${CMAKE_INSTALL_PREFIX}
        variant=release
        runtime-link=shared
    INSTALL_COMMAND "")
