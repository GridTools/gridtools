add_executable(FunctorDoMethodLookupMaps FunctorDoMethodLookupMaps.cpp)
add_test(test.FunctorDoMethodLookupMaps FunctorDoMethodLookupMaps)

add_executable(FunctorDoMethods FunctorDoMethods.cpp)
add_test(test.FunctorDoMethods FunctorDoMethods)

add_executable(LoopIntervals LoopIntervals.cpp)
add_test(test.LoopIntervals LoopIntervals)
add_executable(test-independent test-independent.cpp)
if (NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  set_target_properties(test-independent PROPERTIES COMPILE_FLAGS "-std=c++11")
endif()
add_test(tests.test_independent test-independent)

if (GTEST_FOUND)
  add_executable(tests tests.cpp test_domain_indices.cpp test_smallstorage_indices.cpp boundary_conditions_test.cpp)
  if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    target_link_libraries(tests ${GTEST_BOTH_LIBRARIES})
  else()
    target_link_libraries(tests ${GTEST_BOTH_LIBRARIES} -lpthread)
  endif()
  add_test(test.tests tests)

  if (CUDA_FOUND)
    cuda_add_executable( tests_gpu
      tests.cu
      test_domain.cu
      test_cuda_storage.cu
      test_hybrid_pointer.cu
      gpu_clone.cu
      cloningstuff.cu
      boundary_conditions_test.cu
    )
    target_link_libraries( tests_gpu ${GTEST_BOTH_LIBRARIES} )
    add_test(tests.tests_gpu tests_gpu)

    cuda_add_executable( cloning_derived cloning_derived.cu )
    add_test( tests.cloning_derived cloning_derived 42 )
  endif()
endif()

add_subdirectory( communication )
