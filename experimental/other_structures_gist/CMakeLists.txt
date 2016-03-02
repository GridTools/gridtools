cmake_minimum_required(VERSION 2.8)
project(other_grid_structures CXX)

### set(Boost_USE_STATIC_LIBS OFF)
### set(Boost_USE_MULTITHREADED ON)
### find_package(Boost REQUIRED)
### include_directories("${Boost_INCLUDE_DIRS}")

if(MSVC)
  set(CMAKE_CXX_FLAGS "/EHsc")
  add_definitions( -D_SCL_SECURE_NO_WARNINGS )
endif(MSVC)

add_executable(triangular_2D triangular_2D.cpp)
target_link_libraries(triangular_2D ${Boost_LIBRARIES})

add_executable(cartesian_3D cartesian_3D.cpp)
target_link_libraries(cartesian_3D ${Boost_LIBRARIES})

add_executable(sphered_cube sphered_cube.cpp)
target_link_libraries(sphered_cube ${Boost_LIBRARIES})

