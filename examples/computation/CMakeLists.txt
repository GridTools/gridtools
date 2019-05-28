cmake_minimum_required(VERSION 3.14)
project(asdf)

find_package(GridTools 1.1.0 HINTS ../../install/lib/cmake)

add_executable(driver driver.cpp interpolate_stencil.cpp)
target_link_libraries(driver GridTools::gridtools)
