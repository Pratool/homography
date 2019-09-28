cmake_minimum_required(VERSION 3.8.2)

find_package(OpenCV 4.1. REQUIRED)

add_executable(testpanorama
    ${CMAKE_CURRENT_SOURCE_DIR}/PanoramaStitcher.cpp)

target_link_libraries(testpanorama
    ${OpenCV_LIBS}
    Panorama)

install(TARGETS testpanorama
    DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)



find_package(Eigen3 REQUIRED)
find_package(GTest 1.8 REQUIRED)

add_executable(TestPanoramaUtility
    ${CMAKE_CURRENT_SOURCE_DIR}/TestUtility.cpp)

target_link_libraries(TestPanoramaUtility
    ${OpenCV_LIBS}
    Eigen3::Eigen
    GTest::GTest
    Panorama)

install(TARGETS TestPanoramaUtility
    DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
