add_library(GridToolsExample INTERFACE)
target_compile_definitions(GridToolsExample INTERFACE STRUCTURED_GRIDS)

add_bindings_library(copy_stencil SOURCES copy_stencil_wrapper.cpp)
target_link_libraries(copy_stencil PUBLIC GridToolsExample)

if(GT_ENABLE_TARGET_MC)
    if (CMAKE_C_COMPILER_LOADED)
        add_executable(example_driver_mc_c driver_mc.c)
        target_link_libraries(example_driver_mc_c copy_stencil_c)

        install_example(example_driver_mc_c)
    endif()

    if (CMAKE_Fortran_COMPILER_LOADED)
        include(fortran_helpers)

        add_executable(example_driver_fortran driver.f90)
        target_link_libraries(example_driver_fortran copy_stencil_fortran)
        set_target_properties(example_driver_fortran PROPERTIES LINKER_LANGUAGE Fortran)
        enable_fortran_preprocessing_on_target(example_driver_fortran)

        install_example(example_driver_fortran)
    endif()
endif(GT_ENABLE_TARGET_MC)

if(GT_ENABLE_TARGET_CUDA)
    if (CMAKE_C_COMPILER_LOADED)
        add_executable(example_driver_cu_c driver_cu.c)
        target_link_libraries(example_driver_cu_c copy_stencil_c)

        install_example(example_driver_cu_c)
    endif()
endif(GT_ENABLE_TARGET_CUDA)
