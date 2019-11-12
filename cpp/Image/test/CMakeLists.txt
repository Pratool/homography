cmake_minimum_required(VERSION 3.8.2)

add_executable(testimage
    ${CMAKE_CURRENT_SOURCE_DIR}/TestImage.cpp)

target_link_libraries(testimage
    Image)

install(TARGETS testimage
    DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
