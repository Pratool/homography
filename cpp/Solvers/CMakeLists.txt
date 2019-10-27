cmake_minimum_required(VERSION 3.8.2)

find_package(Eigen3 REQUIRED)
find_package(OpenCV 4.1 REQUIRED)

add_library(Solvers INTERFACE)

set_property(TARGET Solvers APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

install(TARGETS Solvers EXPORT SolversConfig
        DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)

install(EXPORT SolversConfig
        DESTINATION share/Solvers/cmake)

export(TARGETS Solvers
       FILE SolversConfig.cmake)

add_subdirectory(test)
