cmake_minimum_required(VERSION 3.8.2)

find_package(GTest 1.8 REQUIRED)

add_executable(TestGeometry
    ${CMAKE_CURRENT_SOURCE_DIR}/TestGeometry.cpp)

target_link_libraries(TestGeometry
    Eigen3::Eigen
    GTest::GTest
    Geometry)

install(TARGETS TestGeometry
    DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
