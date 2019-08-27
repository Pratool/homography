cmake_minimum_required(VERSION 3.8.2)

find_package(OpenCV 4.1. REQUIRED)

add_executable(testpanorama
    ${CMAKE_CURRENT_SOURCE_DIR}/PanoramaStitcher.cpp)

target_link_libraries(testpanorama
    ${OpenCV_LIBS}
    Panorama)

install(TARGETS testpanorama
    DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
