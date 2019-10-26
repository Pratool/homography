cmake_minimum_required(VERSION 3.8.2)

find_package(Eigen3 REQUIRED)
find_package(OpenCV 4.1 REQUIRED)

add_library(Panorama SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/source/Utility.cpp)

target_include_directories(Panorama
    PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/source)

target_link_libraries(Panorama
    ${OpenCV_LIBS}
    Eigen3::Eigen
    Solvers)

install(TARGETS Panorama EXPORT PanoramaConfig
        DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)

install(EXPORT PanoramaConfig
        DESTINATION share/Panorama/cmake)

export(TARGETS Panorama
       FILE PanoramaConfig.cmake)

add_subdirectory(test)
