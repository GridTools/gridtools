include(bindings)

add_bindings_library(implementation implementation.cpp)

add_executable(driver driver.c)
target_include_directories(driver PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
target_link_libraries(driver implementation)
add_dependencies(driver implementation_declarations)

enable_language (Fortran)

set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -cpp")

add_executable(fdriver fdriver.f90 ${CMAKE_CURRENT_BINARY_DIR}/implementation.f90)
target_link_libraries(fdriver implementation)
add_dependencies(fdriver implementation_declarations)

# On tave Intel Fortran is not working
if(CMAKE_Fortran_COMPILER_ID MATCHES "Intel")
    set_target_properties(fdriver PROPERTIES EXCLUDE_FROM_ALL 1)
endif()