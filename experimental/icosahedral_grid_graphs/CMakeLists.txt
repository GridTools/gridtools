cmake_minimum_required(VERSION 2.8)
project(icosahedral_grid_graphs CXX)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

### set(Boost_USE_STATIC_LIBS OFF)
### set(Boost_USE_MULTITHREADED ON)
### find_package(Boost REQUIRED)
include_directories("${Boost_INCLUDE_DIRS}")

add_subdirectory(src)


