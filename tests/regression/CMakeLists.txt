# special target for executables which are used from performance benchmarks
add_custom_target(perftests)

function(gridtools_add_regression_test tgt_name)
    set(options PERFTEST)
    set(one_value_args LIB_PREFIX)
    set(multi_value_args KEYS SOURCES LABELS)
    cmake_parse_arguments(ARGS "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})
    foreach(key IN LISTS ARGS_KEYS)
        foreach(float_type float double)
            set(tgt ${tgt_name}_${key}_${float_type})
            gridtools_add_test_executable(${tgt}
                    SOURCES ${ARGS_SOURCES}
                    LIBRARIES gridtools regression_main ${ARGS_LIB_PREFIX}_${key})
            target_compile_definitions(${tgt} PRIVATE GT_FLOAT_TYPE=${float_type})
            add_test(NAME tests.${tgt}_12_33_61 COMMAND $<TARGET_FILE:${tgt}> 12 33 61)
            add_test(NAME tests.${tgt}_23_11_43 COMMAND $<TARGET_FILE:${tgt}> 23 11 43)
            set(labels regression ${key} ${float_type} ${ARGS_LABELS})
            set_tests_properties(tests.${tgt}_12_33_61 PROPERTIES LABELS "${labels}")
            set_tests_properties(tests.${tgt}_23_11_43 PROPERTIES LABELS "${labels}")
            if (ARGS_PERFTEST AND NOT key STREQUAL naive)
                add_dependencies(perftests ${tgt})
            endif()
        endforeach()
    endforeach()
endfunction()

function(add_backend_testees prefix)
    foreach(backend IN LISTS ARGN)
        set(tgt ${prefix}_${backend})
        add_library(${tgt} INTERFACE)
        target_link_libraries(${tgt} INTERFACE backend_${backend})
        if (backend STREQUAL naive)
            set(u_storage X86)
        else()
            string(TOUPPER ${backend} u_storage)
        endif()
        if(backend STREQUAL cuda)
            set(timer CUDA)
        elseif (backend STREQUAL mc OR backend STREQUAL x86)
            set(timer OMP)
        else()
            set(timer DUMMY)
        endif()
        string(TOUPPER ${backend} u_backend)
        target_compile_definitions(${tgt} INTERFACE
                GT_BACKEND_${u_backend}
                GT_STORAGE_${u_storage}
                GT_TIMER_${timer})
    endforeach()
endfunction()
add_backend_testees(backend_testee ${GT_BACKENDS})

function(gridtools_add_cartesian_regression_test tgt_name)
    gridtools_add_regression_test(${tgt_name} ${ARGN}
            LIB_PREFIX backend_testee
            KEYS ${GT_BACKENDS}
            LABELS cartesian)
endfunction()

function(gridtools_add_layout_transformation_test)
    foreach(arch IN LISTS GT_GCL_ARCHS)
        set(tgt layout_transformation_testee_${arch})
        add_library(${tgt} INTERFACE)
        target_link_libraries(${tgt} INTERFACE layout_transformation_${arch})
        if (arch STREQUAL gpu)
            target_compile_definitions(${tgt} INTERFACE GT_STORAGE_CUDA GT_TIMER_CUDA)
        elseif (arch STREQUAL cpu)
            target_compile_definitions(${tgt} INTERFACE GT_STORAGE_X86 GT_TIMER_OMP)
        endif()
    endforeach()
    gridtools_add_regression_test(layout_transformation
            LIB_PREFIX layout_transformation_testee
            KEYS ${GT_GCL_ARCHS}
            SOURCES layout_transformation.cpp
            PERFTEST)
endfunction()

function(gridtools_add_boundary_conditions_test)
    foreach(arch IN LISTS GT_GCL_ARCHS)
        set(tgt bc_testee_${arch})
        add_library(${tgt} INTERFACE)
        target_link_libraries(${tgt} INTERFACE bc_${arch})
        if (arch STREQUAL gpu)
            target_compile_definitions(${tgt} INTERFACE GT_STORAGE_CUDA GT_TIMER_CUDA GT_GCL_GPU)
        elseif (arch STREQUAL cpu)
            target_compile_definitions(${tgt} INTERFACE GT_STORAGE_X86 GT_TIMER_OMP GT_GCL_CPU)
        endif()
    endforeach()
    gridtools_add_regression_test(boundary_conditions
            LIB_PREFIX bc_testee
            KEYS ${GT_GCL_ARCHS}
            SOURCES boundary_conditions.cpp
            PERFTEST)
endfunction()

function(gridtools_add_copy_stencil_parallel_test)
    if(TARGET MPI::MPI_CXX)
        foreach(arch IN LISTS GT_GCL_ARCHS)
            foreach(float_type float double)
                set(tgt copy_stencil_parallel_${arch}_${float_type})
                if (arch STREQUAL gpu)
                    set(storage cuda)
                    set(backend cuda)
                elseif (arch STREQUAL cpu)
                    set(storage x86)
                    set(backend x86)
                endif()
                gridtools_add_test_executable(${tgt}
                        SOURCES copy_stencil_parallel.cpp
                        LIBRARIES gridtools mpi_gtest_main gcl_${arch} backend_${backend})
                string(TOUPPER ${arch} u_arch)
                string(TOUPPER ${backend} u_backend)
                string(TOUPPER ${storage} u_storage)
                target_compile_definitions(${tgt} PRIVATE
                        GT_FLOAT_TYPE=${float_type}
                        GT_STORAGE_${u_storage}
                        GT_BACKEND_${u_backend}
                        GT_GCL_${u_arch})
                add_test(NAME ${tgt}
                        COMMAND  ${MPITEST_EXECUTABLE} ${MPITEST_NUMPROC_FLAG} 4 ${MPITEST_PREFLAGS} $<TARGET_FILE:${tgt}> ${MPITEST_POSTFLAGS})
                set(labels mpi regression ${arch} ${float_type})
                set_tests_properties(${tgt} PROPERTIES LABELS "${labels}")
                set_tests_properties(${tgt} PROPERTIES PROCESSORS 4)
            endforeach()
        endforeach()
    endif()
endfunction()

gridtools_add_cartesian_regression_test(horizontal_diffusion SOURCES horizontal_diffusion.cpp PERFTEST)
gridtools_add_cartesian_regression_test(horizontal_diffusion_fused SOURCES horizontal_diffusion_fused.cpp PERFTEST)
gridtools_add_cartesian_regression_test(simple_hori_diff SOURCES simple_hori_diff.cpp PERFTEST)
gridtools_add_cartesian_regression_test(copy_stencil SOURCES copy_stencil.cpp PERFTEST)
gridtools_add_cartesian_regression_test(vertical_advection_dycore SOURCES vertical_advection_dycore.cpp PERFTEST)
gridtools_add_cartesian_regression_test(advection_pdbott_prepare_tracers SOURCES advection_pdbott_prepare_tracers.cpp PERFTEST)
gridtools_add_cartesian_regression_test(laplacian SOURCES laplacian.cpp)
gridtools_add_cartesian_regression_test(positional_stencil SOURCES positional_stencil.cpp)
gridtools_add_cartesian_regression_test(tridiagonal SOURCES tridiagonal.cpp)
gridtools_add_cartesian_regression_test(alignment SOURCES alignment.cpp)
gridtools_add_cartesian_regression_test(extended_4D SOURCES extended_4D.cpp)
gridtools_add_cartesian_regression_test(expandable_parameters SOURCES expandable_parameters.cpp)
gridtools_add_cartesian_regression_test(expandable_parameters_single_kernel SOURCES expandable_parameters_single_kernel.cpp)
gridtools_add_cartesian_regression_test(horizontal_diffusion_functions SOURCES horizontal_diffusion_functions.cpp)
gridtools_add_layout_transformation_test()
gridtools_add_boundary_conditions_test()
gridtools_add_copy_stencil_parallel_test()

add_subdirectory(icosahedral)
add_subdirectory(c_bindings)
add_subdirectory(communication)
