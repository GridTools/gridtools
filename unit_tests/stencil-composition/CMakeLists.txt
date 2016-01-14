
include_directories( ../../include/stencil-composition ) ## Library header files

FILE(GLOB_RECURSE INC_ALL "../../include/stencil-composition/[a-z]*.hpp")

SET(STENCIL_COMPOSITION_UNITTEST_SOURCES test_accessor.cpp ${INC_ALL})
if ( ENABLE_CXX11 )
  SET(STENCIL_COMPOSITION_UNITTEST_SOURCES ${STENCIL_COMPOSITION_UNITTEST_SOURCES} test-assign-placeholders.cpp )
endif( ENABLE_CXX11 )

if( ENABLE_CXX11 )
        foreach( variation CALL OFFSETS PROCEDURES )
            string(TOLOWER ${variation} var_suffix)
            add_executable(multi_types_block_${var_suffix} multi_types_test.cpp)
            add_executable(multi_types_naive_${var_suffix} multi_types_test.cpp)
            set_target_properties(multi_types_block_${var_suffix} PROPERTIES COMPILE_FLAGS "${ADDITIONAL_CXX_FLAGS} -DFUNCTIONS_${variation} -DBACKEND_BLOCK")
            set_target_properties(multi_types_naive_${var_suffix} PROPERTIES COMPILE_FLAGS "${ADDITIONAL_CXX_FLAGS} -DFUNCTIONS_${variation}")
            target_link_libraries(multi_types_block_${var_suffix} ${exe_LIBS} gcl ${ADD_MPI_LIBS} ${MPI_ADDITIONAL_LIBS})
            target_link_libraries(multi_types_naive_${var_suffix} ${exe_LIBS} gcl ${ADD_MPI_LIBS} ${MPI_ADDITIONAL_LIBS})
            gridtools_add_test(tests.multi_types_block_${var_suffix} ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/build/multi_types_block_${var_suffix}")
            gridtools_add_test(tests.multi_types_naive_${var_suffix} ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/build/multi_types_naive_${var_suffix}")
        endforeach(variation)
endif()

if(CUDA_FOUND)
     if( ENABLE_CXX11 )
        foreach( variation CALL OFFSETS PROCEDURES )
            string(TOLOWER ${variation} var_suffix)
            cuda_add_executable(multi_types_cuda_${var_suffix} multi_types_test.cu OPTIONS "-DFUNCTIONS_${variation}")
            target_link_libraries(multi_types_cuda_${var_suffix} ${exe_LIBS} gcl ${ADD_MPI_LIBS} ${MPI_ADDITIONAL_LIBS})
            gridtools_add_test(tests.multi_types_cuda_${var_suffix} ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/build/multi_types_cuda_${var_suffix}")
        endforeach(variation)
     endif()
endif()

### Executables
#CPU tests
add_executable(stencil_composition_tests ${STENCIL_COMPOSITION_UNITTEST_SOURCES})
gridtools_add_test(tests.stencil_composition ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/build/stencil_composition_tests")

set_target_properties(stencil_composition_tests PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS})
target_link_libraries(stencil_composition_tests ${GTEST_BOTH_LIBRARIES} -lpthread ${exe_LIBS})

add_subdirectory(caches)
if (NOT STRUCTURED_GRIDS)
  add_subdirectory(icosahedral_grids)
else()
  add_subdirectory(structured_grids)
endif()
