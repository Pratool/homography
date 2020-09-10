find_package(Boost REQUIRED)
add_executable(renderd main.cpp)
target_link_libraries(renderd Boost::boost)
install(TARGETS renderd RUNTIME DESTINATION bin)
