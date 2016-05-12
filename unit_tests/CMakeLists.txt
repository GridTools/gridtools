#add_executable(FunctorDoMethodLookupMaps FunctorDoMethodLookupMaps.cpp)
#gridtools_add_test(test.FunctorDoMethodLookupMaps ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/build/FunctorDoMethodLookupMaps")

#add_executable(FunctorDoMethods FunctorDoMethods.cpp)
#gridtools_add_test(test.FunctorDoMethods ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/build/FunctorDoMethods")

#add_executable(LoopIntervals LoopIntervals.cpp)
#gridtools_add_test(test.LoopIntervals ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/build/LoopIntervals")

if ( ENABLE_CXX11 AND STRUCTURED_GRIDS)
    add_executable(test-independent test-independent.cpp)
    set_target_properties(test-independent PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS})
    target_link_libraries(test-independent ${exe_LIBS})
    gridtools_add_test(tests.test_independent ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/build/test-independent")

endif()

set(COMMON_TESTS
    test_domain_indices.cpp
    boundary_conditions_test.cpp
)

if(STRUCTURED_GRIDS)
    #set(COMMON_TESTS ${COMMON_TESTS} tests.cpp test_global_accessor.cpp)
endif(STRUCTURED_GRIDS)

add_executable(tests ${COMMON_TESTS})

if(STRUCTURED_GRIDS)
#add_executable(test_global_accessor test_global_accessor.cpp) #does not compile
#set_target_properties(test_global_accessor PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS})
#target_link_libraries(test_global_accessor ${GTEST_BOTH_LIBRARIES} gcl -lpthread ${exe_LIBS})
endif(STRUCTURED_GRIDS)

# include boost
include_directories( SYSTEM ${Boost_INCLUDE_DIRS} )
link_directories( ${Boost_LIBRARY_DIRS} )

set_target_properties(tests PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS})

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    target_link_libraries(tests ${GTEST_BOTH_LIBRARIES} gcl ${exe_LIBS})
else()
    target_link_libraries(tests ${GTEST_BOTH_LIBRARIES} gcl -lpthread ${exe_LIBS})
endif()


gridtools_add_test(test.tests ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/build/tests")

if (CUDA_FOUND AND STRUCTURED_GRIDS)
    set(GPU_SOURCES tests.cu
      test_domain.cu
      test_cuda_storage.cu
      test_hybrid_pointer.cu
      gpu_clone.cu
      cloningstuff.cu
    )
    if(STRUCTURED_GRIDS)
      set(GPU_SOURCES ${GPU_SOURCES}
        boundary_conditions_test.cu
        test_global_accessor.cu
      )
    endif(STRUCTURED_GRIDS)

    cuda_add_executable( tests_gpu
      ${GPU_SOURCES}
    )

    gridtools_add_test(tests.tests_gpu ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/build/tests_gpu")

    target_link_libraries(tests_gpu ${GTEST_BOTH_LIBRARIES} gcl ${exe_LIBS})

    cuda_add_executable( cloning_derived cloning_derived.cu )
    target_link_libraries(cloning_derived ${GTEST_BOTH_LIBRARIES} ${exe_LIBS})
    gridtools_add_test( tests.cloning_derived ${TEST_SCRIPT} "${CMAKE_BINARY_DIR}/build/cloning_derived" 42)

    set(ENV{BIN_DIR} ${CMAKE_BINARY_DIR}/build)
    set(ENV{SRC_DIR} ${CMAKE_SOURCE_DIR})

    if(GNU_COVERAGE)
      if(GCOVR_PATH)
        set( GCOVR_PATH ${GCOVR_PATH}/ )
      endif( GCOVR_PATH )
      add_custom_command(TARGET tests
        POST_BUILD
        COMMAND echo "running the coverage sampling"
        COMMAND $ENV{BIN_DIR}/tests
        COMMAND ${GCOVR_PATH}gcovr  -r $ENV{SRC_DIR}
        COMMAND ${GCOVR_PATH}gcovr  --branches -r $ENV{SRC_DIR}
      )
    endif()

endif()

if (CUDA_FOUND)
   cuda_add_executable( test_cuda_storage test_cuda_storage_exe.cu )
   target_link_libraries(test_cuda_storage ${GTEST_BOTH_LIBRARIES} ${exe_LIBS})
endif()

if(STRUCTURED_GRIDS)
    add_executable(CopyStencil external_ptr_test/CopyStencil.cpp)
    set_target_properties(CopyStencil PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS})
    target_link_libraries(CopyStencil ${exe_LIBS})

    add_subdirectory( communication )
endif(STRUCTURED_GRIDS)

#add_subdirectory( common )
#add_subdirectory( stencil-composition )
#add_subdirectory( storage )
add_subdirectory( reductions )
