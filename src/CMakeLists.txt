set(CMAKE_CXX_STANDARD 17)

file(GLOB PROJECT_CPP *.cpp search/*.cpp utils/*.cpp)
add_library(${PROJECT_NAME} ${PROJECT_CPP})
