cmake_minimum_required(VERSION 3.8.2)

find_package(OpenCV 4.1 REQUIRED)

add_library(Panorama SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/source/Utility.cpp)

target_include_directories(Panorama
    PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/source)

target_link_libraries(Panorama ${OpenCV_LIBS})

install(TARGETS Panorama EXPORT Panorama
        DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)

add_subdirectory(test)
