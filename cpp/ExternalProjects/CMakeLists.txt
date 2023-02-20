cmake_minimum_required(VERSION 3.6)

project(ExternalProjects)

# This will build a subset of dependencies often found at operating-system level
# package managers (e.g. Debian packages managed by aptitude). This file copies
# much of the CMake external dependencies written by ajakhotia in his robotFarm
# repository found here: https://github.com/ajakhotia/robotFarm

include(ExternalProject)

externalproject_add(Eigen3ExternalProject
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/eigen3
    URL "https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.zip"
    URL_HASH SHA256=e09b89aae054e9778ee3f606192ee76d645eec82c402c01c648b1fe46b6b9857
    DOWNLOAD_NO_PROGRESS ON
    CMAKE_ARGS
        -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
        -DCMAKE_PREFIX_PATH:PATH=${CMAKE_INSTALL_PREFIX}
        -DCMAKE_INSTALL_RPATH:PATH=$ORIGIN:$ORIGIN/../lib
        -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_INSTALL_PREFIX})


externalproject_add(GoogleTestExternalProject
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/googleTest
    URL "https://github.com/google/googletest/archive/release-1.8.1.zip"
    URL_HASH SHA256=927827c183d01734cc5cfef85e0ff3f5a92ffe6188e0d18e909c5efebf28a0c7
    DOWNLOAD_NO_PROGRESS ON
    CMAKE_ARGS
        -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
        -DCMAKE_PREFIX_PATH:PATH=${CMAKE_INSTALL_PREFIX}
        -DCMAKE_INSTALL_RPATH:PATH=$ORIGIN:$ORIGIN/../lib
        -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_INSTALL_PREFIX})


externalproject_add(GlogExternalProject
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/glog
    URL "https://github.com/google/glog/archive/v0.4.0.zip"
    URL_HASH SHA256=9e1b54eb2782f53cd8af107ecf08d2ab64b8d0dc2b7f5594472f3bd63ca85cdc
    DOWNLOAD_NO_PROGRESS ON
    CMAKE_ARGS
        -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
        -DCMAKE_PREFIX_PATH:PATH=${CMAKE_INSTALL_PREFIX}
        -DCMAKE_INSTALL_RPATH:PATH=$ORIGIN:$ORIGIN/../lib
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
        -DCMAKE_INSTALL_RPATH:PATH=$ORIGIN:$ORIGIN/../lib
        -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_INSTALL_PREFIX})

add_dependencies(CeresSolverExternalProject GlogExternalProject Eigen3ExternalProject)


externalproject_add(PythonExternalProject
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/python
    URL "https://github.com/python/cpython/archive/v3.8.5.zip"
    URL_HASH SHA256=505cf91db10dccbccc4b158c208d0c729b05aa60aa6e82c27f1c1bf2b053cef2
    DOWNLOAD_NO_PROGRESS ON
    CONFIGURE_COMMAND <SOURCE_DIR>/configure
        --prefix=${CMAKE_INSTALL_PREFIX}
        --enable-shared
        --enable-optimizations
        --enable-loadable-sqlite-extensions
        --disable-ipv6
        --enable-big-digits=30
        --with-lto
        --with-system-expat
        --with-system-ffi
        --with-system-libmpdec
        --with-signal-module
        --with-threads
        --with-pymalloc
        --with-fpectl
        --with-computed-gotos
        --with-ensurepip=upgrade)

externalproject_add(OpenCVExternalProject
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/opencv
    URL "https://github.com/opencv/opencv/archive/4.1.1.zip"
    URL_HASH SHA256=d34985c7b4283519c032e3585b30846644f56c81acba35693295ff8930f080aa
    DOWNLOAD_NO_PROGRESS ON
    CONFIGURE_COMMAND
        LD_LIBRARY_PATH=${CMAKE_INSTALL_PREFIX}/lib:$ENV{LD_LIBRARY_PATH}
        PATH=${CMAKE_INSTALL_PREFIX}/bin:$ENV{PATH}
        ${CMAKE_COMMAND} <SOURCE_DIR>
        -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
        -DCMAKE_PREFIX_PATH:PATH=${CMAKE_INSTALL_PREFIX}
        -DCMAKE_INSTALL_RPATH:PATH=$ORIGIN:$ORIGIN/../lib
        -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_INSTALL_PREFIX}
        -DBUILD_TESTS:BOOL=OFF
        -DWITH_VTK:BOOL=ON
        -DWITH_NVCUVID:BOOL=OFF)

add_dependencies(OpenCVExternalProject PythonExternalProject)


externalproject_add(ZlibExternalProject
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/zlib
    URL "https://www.zlib.net/zlib-1.2.13.tar.gz"
    URL_HASH SHA256=b3a24de97a8fdbc835b9833169501030b8977031bcb54b3b3ac13740f846ab30
    DOWNLOAD_NO_PROGRESS ON
    CMAKE_ARGS
        -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
        -DCMAKE_PREFIX_PATH:PATH=${CMAKE_INSTALL_PREFIX}
        -DCMAKE_INSTALL_RPATH:PATH=$ORIGIN:$ORIGIN/../lib
        -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_INSTALL_PREFIX})

externalproject_add(PngExternalProject
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/png
    URL "https://github.com/glennrp/libpng/archive/refs/tags/v1.6.39.tar.gz"
    URL_HASH SHA256=a00e9d2f2f664186e4202db9299397f851aea71b36a35e74910b8820e380d441
    DOWNLOAD_NO_PROGRESS ON
    CMAKE_ARGS
        -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
        -DCMAKE_PREFIX_PATH:PATH=${CMAKE_INSTALL_PREFIX}
        -DCMAKE_INSTALL_RPATH:PATH=$ORIGIN:$ORIGIN/../lib
        -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_INSTALL_PREFIX})
add_dependencies(PngExternalProject ZlibExternalProject)

externalproject_add(BoostExternalProject
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/boost
    URL "https://boostorg.jfrog.io/artifactory/main/release/1.72.0/source/boost_1_72_0.tar.bz2"
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
        hardcode-dll-paths=true
        dll-path="'\$ORIGIN/../lib'"
        --prefix=${CMAKE_INSTALL_PREFIX}
		--with-filesystem
		--with-headers
		--with-iostreams
		--with-program_options
		--with-system
        variant=release
        runtime-link=shared
    INSTALL_COMMAND "")

add_dependencies(BoostExternalProject PythonExternalProject)
add_dependencies(BoostExternalProject PngExternalProject)
