if( CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR )

    set(STANDALONE ON)

    # for simplicity of this CMakeLists, we only enable the mc target in the standalone
    set(GT_ENABLE_TARGET_MC ON)

    cmake_minimum_required(VERSION 3.12.4)

    project(GridTools_CBindings)

    set(CMAKE_CXX_STANDARD 11)

    find_package(GridTools REQUIRED)
    list(APPEND CMAKE_MODULE_PATH "${GridTools_MODULE_PATH}")

    # use the structured grid interface
    add_library(GridToolsExample INTERFACE)
    target_compile_definitions(GridToolsExample INTERFACE GT_STRUCTURED_GRIDS)
    target_link_libraries(GridToolsExample INTERFACE GridTools::gridtools)

    include(gt_bindings)

else()

    set(STANDALONE OFF)

    add_library(GridToolsExample INTERFACE)
    target_link_libraries(GridToolsExample INTERFACE GridToolsTest)

endif()

if(GT_ENABLE_TARGET_MC)
    # our bindings library is different for mc and cuda-backend. This CCMake function
    # generates two targets, once suffixed with _c and once suffixed with _fortran.
    gt_add_bindings_library(copy_stencil_lib_mc SOURCES copy_stencil_wrapper.cpp)
    target_link_libraries(copy_stencil_lib_mc PUBLIC GridToolsExample)

    if (CMAKE_C_COMPILER_LOADED)
        add_executable(example_driver_mc_c driver_mc.c)
        target_link_libraries(example_driver_mc_c copy_stencil_lib_mc_c)

        if (NOT STANDALONE)
            install_example(example_driver_mc_c)
        endif()
    endif()

    if (CMAKE_Fortran_COMPILER_LOADED)
        include(fortran_helpers)

        add_executable(example_driver_fortran driver.f90)
        target_link_libraries(example_driver_fortran copy_stencil_lib_mc_fortran)
        set_target_properties(example_driver_fortran PROPERTIES LINKER_LANGUAGE Fortran)
        # this helper enables the preprocessor by setting a compiler-dependent flag
        gt_enable_fortran_preprocessing_on_target(example_driver_fortran)

        if (NOT STANDALONE)
            install_example(example_driver_fortran)
        endif()
    endif()
endif(GT_ENABLE_TARGET_MC)

if(GT_ENABLE_TARGET_CUDA)
    add_bindings_library(copy_stencil_lib_cu SOURCES copy_stencil_wrapper.cu)
    target_link_libraries(copy_stencil_lib_cu PUBLIC GridToolsExample)

    if (CMAKE_C_COMPILER_LOADED)
        add_executable(example_driver_cu_c driver_cu.c)
        target_link_libraries(example_driver_cu_c copy_stencil_lib_cu_c)

        if (NOT STANDALONE)
            install_example(example_driver_cu_c)
        endif()
    endif()
endif(GT_ENABLE_TARGET_CUDA)
