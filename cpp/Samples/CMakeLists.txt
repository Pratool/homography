cmake_minimum_required(VERSION 3.8.2)

find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED)
find_package(PNG REQUIRED)

add_executable(generateGradient source/generateGradient.cpp)

target_link_libraries(generateGradient
  Boost::boost
  Eigen3::Eigen
  PNG::PNG
)

install(
  TARGETS generateGradient
  DESTINATION bin
)
