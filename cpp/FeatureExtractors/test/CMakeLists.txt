find_package(PNG REQUIRED)
find_package(TIFF REQUIRED)

add_executable(GradientTest GradientTest.cpp)

target_link_libraries(GradientTest
  PNG::PNG
  TIFF::TIFF
  FeatureExtractors
)
