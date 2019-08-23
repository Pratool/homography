cmake_minimum_required(VERSION 3.5.2)

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
