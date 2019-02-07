add_bindings_library(copy_stencil SOURCES copy_stencil_wrapper.cpp)
target_link_libraries(copy_stencil PUBLIC GridToolsTestX86)

if (CMAKE_C_COMPILER_LOADED)
    add_executable(example_driver_c driver.c)
    target_link_libraries(example_driver_c copy_stencil_c)
endif()

if (CMAKE_Fortran_COMPILER_LOADED)
    include(fortran_helpers)

    add_executable(example_driver_fortran driver.f90)
    target_link_libraries(example_driver_fortran copy_stencil_fortran)
    set_target_properties(example_driver_fortran PROPERTIES LINKER_LANGUAGE Fortran)
    enable_fortran_preprocessing_on_target(example_driver_fortran)
endif()


