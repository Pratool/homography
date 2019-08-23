cmake_minimum_required(VERSION 3.8.2)

find_package(OpenCV 4.1 REQUIRED)

add_executable(calibrator
    ${CMAKE_CURRENT_SOURCE_DIR}/source/CameraCalibration.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/Calibrator.cpp)

target_include_directories(calibrator PRIVATE include)

target_link_libraries(calibrator ${OpenCV_LIBS})

install(TARGETS calibrator
        DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
