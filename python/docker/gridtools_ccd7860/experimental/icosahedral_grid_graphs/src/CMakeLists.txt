add_executable(triangular_2D triangular_2D.cpp)
target_link_libraries(triangular_2D ${Boost_LIBRARIES})

add_subdirectory(test)
