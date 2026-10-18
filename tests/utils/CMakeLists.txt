set(CMAKE_CXX_STANDARD 17)

add_executable(gen_random_slice gen_random_slice.cpp)
target_link_libraries(gen_random_slice ${PROJECT_NAME})
