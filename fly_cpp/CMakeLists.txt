cmake_minimum_required(VERSION 3.16)
project(fly_cpp CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
find_package(OpenMP)
include_directories(include)
add_library(flycore
  src/core.cpp
  src/sensors.cpp
  src/clock.cpp
  src/xzone.cpp
  src/spike.cpp
  src/battery.cpp
)
if(OpenMP_CXX_FOUND)
  target_link_libraries(flycore PUBLIC OpenMP::OpenMP_CXX)
endif()
add_executable(fly src/main.cpp)
target_link_libraries(fly PRIVATE flycore)
