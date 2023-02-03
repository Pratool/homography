cmake_minimum_required(VERSION 3.8.2)

find_package(OpenCV 4.1 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(GTest 1.8 REQUIRED)

add_executable(TestSolvers
    ${CMAKE_CURRENT_SOURCE_DIR}/TestSolvers.cpp)

target_link_libraries(TestSolvers
    ${OpenCV_LIBS}
    Eigen3::Eigen
    GTest::GTest
    Solvers)

install(TARGETS TestSolvers
    DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
