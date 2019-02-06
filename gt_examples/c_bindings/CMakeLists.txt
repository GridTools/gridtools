add_bindings_library(copy_stencil SOURCES copy_stencil.cpp)
target_link_libraries(copy_stencil PUBLIC GridToolsTestX86)

if (CMAKE_Fortran_COMPILER_LOADED)
    include(fortran_helpers)

    add_executable(example_driver driver.f90)
    target_link_libraries(example_driver copy_stencil_fortran)
    set_target_properties(example_driver PROPERTIES LINKER_LANGUAGE Fortran)
    enable_fortran_preprocessing_on_target(example_driver)
endif()


