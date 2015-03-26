add_executable(FunctorDoMethodLookupMaps FunctorDoMethodLookupMaps.cpp)
add_test(test.FunctorDoMethodLookupMaps "${CMAKE_BINARY_DIR}/build/FunctorDoMethodLookupMaps")

add_executable(FunctorDoMethods FunctorDoMethods.cpp)
add_test(test.FunctorDoMethods "${CMAKE_BINARY_DIR}/build/FunctorDoMethods")

add_executable(LoopIntervals LoopIntervals.cpp)
add_test(test.LoopIntervals "${CMAKE_BINARY_DIR}/build/LoopIntervals")

if ( ENABLE_CXX11 )
    add_executable(test-independent test-independent.cpp)
    set_target_properties(test-independent PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS})
    target_link_libraries(test-independent ${exe_LIBS})
    add_test(tests.test_independent "${CMAKE_BINARY_DIR}/build/test-independent")
endif()

if (GTEST_FOUND)
    if ( ENABLE_CXX11 )
        add_executable(tests tests.cpp test-assign-placeholders.cpp test_domain_indices.cpp test_smallstorage_indices.cpp boundary_conditions_test.cpp)
    else()
        add_executable(tests tests.cpp test_domain_indices.cpp test_smallstorage_indices.cpp boundary_conditions_test.cpp)
    endif()
  # include boost
  include_directories( SYSTEM ${Boost_INCLUDE_DIRS} )
  link_directories( ${Boost_LIBRARY_DIRS} )

  set_target_properties(tests PROPERTIES COMPILE_FLAGS ${ADDITIONAL_CXX_FLAGS})

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

    add_test(tests.tests_gpu "${CMAKE_BINARY_DIR}/build/tests_gpu")

    target_link_libraries(tests_gpu ${GTEST_BOTH_LIBRARIES} ${exe_LIBS})

    cuda_add_executable( cloning_derived cloning_derived.cu )
    target_link_libraries(cloning_derived ${GTEST_BOTH_LIBRARIES} ${exe_LIBS})
    add_test( tests.cloning_derived "${CMAKE_BINARY_DIR}/build/cloning_derived" 42 )

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
endif()

add_subdirectory( communication )
add_subdirectory( common )
add_subdirectory( stencil-composition )

if(PYTHONLIBS_FOUND AND PYTHONINTERP_FOUND)
add_custom_target(python_tests)
  message("enabling Python unit tests")
  message("You should set your PYTHONPATH environment variable to " ${PYTHON_SITE_PACKAGES})

  add_custom_command(TARGET python_tests
		     POST_BUILD
		     COMMAND ${CMAKE_SOURCE_DIR}/python/python_run_tests.sh ${PYTHON_SITE_PACKAGES} ${CMAKE_SOURCE_DIR} ${PYTHON_EXECUTABLE} ${PYTHON_INSTALL_PREFIX}
		     COMMENT "Running Python automatic tests" VERBATIM
		   )
endif(PYTHONLIBS_FOUND AND PYTHONINTERP_FOUND)
