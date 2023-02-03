cmake_minimum_required(VERSION 3.8.2)

find_package(Boost REQUIRED)

add_library(FeatureExtractors 
  source/ImageGradient.cpp)

target_include_directories(FeatureExtractors
    PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        source)

target_link_libraries(FeatureExtractors
  Boost::boost)

install(TARGETS FeatureExtractors EXPORT FeatureExtractorsConfig
        DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)

install(EXPORT FeatureExtractorsConfig
        DESTINATION share/FeatureExtractors/cmake)

export(TARGETS FeatureExtractors
       FILE FeatureExtractorsConfig.cmake)

add_subdirectory(test)
