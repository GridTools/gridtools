# we set up a generic target that holds the options required for all backends
add_library(GridToolsExample INTERFACE)
target_compile_definitions(GridToolsExample INTERFACE GT_STRUCTURED_GRIDS)
target_compile_options(GridToolsExample INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:-arch=${GT_CUDA_ARCH}>)
target_link_libraries(GridToolsExample INTERFACE gridtools)

include(workaround_cuda)
_workaround_cuda(GridToolsExample)

if(GT_ENABLE_TARGET_MC)
    # our bindings library is different for mc and cuda-backend. This CCMake function
    # generates two targets, once suffixed with _c and once suffixed with _fortran.
    add_bindings_library(copy_stencil_lib_mc SOURCES copy_stencil_wrapper.cpp)
    target_link_libraries(copy_stencil_lib_mc PUBLIC GridToolsExample)

    if (CMAKE_C_COMPILER_LOADED)
        add_executable(example_driver_mc_c driver_mc.c)
        target_link_libraries(example_driver_mc_c copy_stencil_lib_mc_c)

        install_example(example_driver_mc_c)
    endif()

    if (CMAKE_Fortran_COMPILER_LOADED)
        include(fortran_helpers)

        add_executable(example_driver_fortran driver.f90)
        target_link_libraries(example_driver_fortran copy_stencil_lib_mc_fortran)
        set_target_properties(example_driver_fortran PROPERTIES LINKER_LANGUAGE Fortran)
        # this helper enables the preprocessor by setting a compiler-dependent flag
        enable_fortran_preprocessing_on_target(example_driver_fortran)

        install_example(example_driver_fortran)
    endif()
endif(GT_ENABLE_TARGET_MC)

if(GT_ENABLE_TARGET_CUDA)
    add_bindings_library(copy_stencil_lib_cu SOURCES copy_stencil_wrapper.cu)
    target_link_libraries(copy_stencil_lib_cu PUBLIC GridToolsExample)

    if (CMAKE_C_COMPILER_LOADED)
        add_executable(example_driver_cu_c driver_cu.c)
        target_link_libraries(example_driver_cu_c copy_stencil_lib_cu_c)

        install_example(example_driver_cu_c)
    endif()
endif(GT_ENABLE_TARGET_CUDA)
