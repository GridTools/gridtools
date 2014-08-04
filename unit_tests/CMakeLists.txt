add_executable(FunctorDoMethodLookupMaps FunctorDoMethodLookupMaps.cpp)
add_test(test.FunctorDoMethodLookupMaps "${CMAKE_BINARY_DIR}/build/FunctorDoMethodLookupMaps")

add_executable(FunctorDoMethods FunctorDoMethods.cpp)
add_test(test.FunctorDoMethods "${CMAKE_BINARY_DIR}/build/FunctorDoMethods")

add_executable(LoopIntervals LoopIntervals.cpp)
add_test(test.LoopIntervals "${CMAKE_BINARY_DIR}/build/LoopIntervals")
add_executable(test-independent test-independent.cpp)
if (NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  set_target_properties(test-independent PROPERTIES COMPILE_FLAGS "-std=c++11")
endif()
add_test(tests.test_independent "${CMAKE_BINARY_DIR}/build/test-independent")

if (GTEST_FOUND)
  add_executable(tests tests.cpp test_domain_indices.cpp test_smallstorage_indices.cpp boundary_conditions_test.cpp)
  # include boost
  include_directories( SYSTEM ${Boost_INCLUDE_DIRS} )
  link_directories( ${Boost_LIBRARY_DIRS} )

  if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    target_link_libraries(tests ${GTEST_BOTH_LIBRARIES} ${exe_LIBS})
  else()
    target_link_libraries(tests ${GTEST_BOTH_LIBRARIES} -lpthread ${exe_LIBS})
  endif()


  add_test(test.tests "${CMAKE_BINARY_DIR}/build/tests")

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
    add_test(tests.tests_gpu "${CMAKE_BINARY_DIR}/build/tests_gpu")

    cuda_add_executable( cloning_derived cloning_derived.cu )
    add_test( tests.cloning_derived "${CMAKE_BINARY_DIR}/build/cloning_derived" 42 )

    set(ENV{BIN_DIR} ${CMAKE_BINARY_DIR}/build)
    set(ENV{SRC_DIR} ${CMAKE_SOURCE_DIR})

if(GNU_COVERAGE)
  add_custom_command(TARGET tests
    POST_BUILD
    COMMAND echo "running the coverage sampling"
    COMMAND $ENV{BIN_DIR}/tests
    COMMAND gcovr  -r $ENV{SRC_DIR}
    COMMAND gcovr  --branches -r $ENV{SRC_DIR}
    )
endif()

  endif()
endif()

add_subdirectory( communication )
