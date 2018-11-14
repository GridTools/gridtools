function(_workaround_icc)
    if( GT_ENABLE_TARGET_CUDA )
      # workaround for boost::optional with CUDA9.2
      # TODO Note that if you need to compile with CUDA 9.2, you cannot build the library with CUDA 9.0!
      # We should fix that by putting this logic into a header.
      if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL "9.2" AND ${CMAKE_CUDA_COMPILER_VERSION} VERSION_LESS "10.1")
          target_compile_definitions(GridTools INTERFACE BOOST_OPTIONAL_CONFIG_USE_OLD_IMPLEMENTATION_OF_OPTIONAL)
          target_compile_definitions(GridTools INTERFACE BOOST_OPTIONAL_USE_OLD_DEFINITION_OF_NONE)
      endif()
    endif()

    # fix buggy Boost MPL config for Intel compiler (last confirmed with Boost 1.67 and ICC 18)
    # otherwise we run into this issue: https://software.intel.com/en-us/forums/intel-c-compiler/topic/516083
    target_compile_definitions(GridTools INTERFACE
        $<$<AND:$<CXX_COMPILER_ID:Intel>,$<VERSION_LESS:$<CXX_COMPILER_VERSION>,19>>:BOOST_MPL_AUX_CONFIG_GCC_HPP_INCLUDED>)
    target_compile_definitions(GridTools INTERFACE
        "$<$<AND:$<CXX_COMPILER_ID:Intel>,$<VERSION_LESS:$<CXX_COMPILER_VERSION>,19>>:BOOST_MPL_CFG_GCC='((__GNUC__ << 8) | __GNUC_MINOR__)'>" )

    # force boost to use decltype() for boost::result_of, required to compile without errors (ICC 17+18)
    target_compile_definitions(GridTools INTERFACE
        $<$<AND:$<CXX_COMPILER_ID:Intel>,$<VERSION_LESS:$<CXX_COMPILER_VERSION>,19>>:BOOST_RESULT_OF_USE_DECLTYPE>)

endfunction()
