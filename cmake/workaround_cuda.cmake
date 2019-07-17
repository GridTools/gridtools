function(_workaround_cuda target)
    if( GT_ENABLE_BACKEND_CUDA AND NOT GT_USE_CLANG_CUDA)
      # workaround for boost::optional with CUDA9.2
      # TODO Note that if you need to compile with CUDA 9.2, you cannot build the library with CUDA 9.0!
      # We should fix that by putting this logic into a header.
      if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL "9.2" AND ${CMAKE_CUDA_COMPILER_VERSION} VERSION_LESS "10.1")
          target_compile_definitions(${target} INTERFACE BOOST_OPTIONAL_CONFIG_USE_OLD_IMPLEMENTATION_OF_OPTIONAL)
          target_compile_definitions(${target} INTERFACE BOOST_OPTIONAL_USE_OLD_DEFINITION_OF_NONE)
      endif()
    endif()
endfunction()

