cmake_minimum_required(VERSION 3.8.2)

find_package(Eigen3 REQUIRED)
find_package(OpenCV 4.1 REQUIRED)

add_library(Image SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/source/Image.cpp)

target_include_directories(Image
    PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/source)

target_link_libraries(Image
    ${OpenCV_LIBS}
    Eigen3::Eigen
    Geometry)

install(TARGETS Image EXPORT ImageConfig
        DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)

install(EXPORT ImageConfig
        DESTINATION share/Image/cmake)

export(TARGETS Image
       FILE ImageConfig.cmake)

add_subdirectory(test)
