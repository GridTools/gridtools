set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR};${CMAKE_MODULE_PATH}")
find_package( NetCDF REQUIRED )

include_directories(${NETCDF_INCLUDES} )

set( GTEST_FILES ../Options.hpp ../Options.cpp ../icosahedral/unstructured_grid.cpp)

add_definitions("-DBENCHMARK")

add_executable(operator_examples operator_examples.cpp IconToGridTools.cpp ${GTEST_FILES})
set_target_properties(operator_examples PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS})
target_link_libraries(operator_examples ${exe_LIBS} gcl ${NETCDF_LIBRARIES} ${NETCDF_LIBRARIES_CXX})
