cmake_minimum_required(VERSION 3.8.2)

find_package(Eigen3 REQUIRED)

add_library(Geometry INTERFACE)

set_property(TARGET Geometry APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

install(TARGETS Geometry EXPORT GeometryConfig
        DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)

install(EXPORT GeometryConfig
        DESTINATION share/Geometry/cmake)

export(TARGETS Geometry
       FILE GeometryConfig.cmake)

add_subdirectory(test)
