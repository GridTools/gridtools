function(_workaround_icc target)
    # fix buggy Boost MPL config for Intel compiler (last confirmed with Boost 1.67 and ICC 18)
    # otherwise we run into this issue: https://software.intel.com/en-us/forums/intel-c-compiler/topic/516083
    target_compile_definitions(${target} INTERFACE
        $<$<AND:$<CXX_COMPILER_ID:Intel>,$<VERSION_LESS:$<CXX_COMPILER_VERSION>,19>>:BOOST_MPL_AUX_CONFIG_GCC_HPP_INCLUDED>)
    target_compile_definitions(${target} INTERFACE
        "$<$<AND:$<CXX_COMPILER_ID:Intel>,$<VERSION_LESS:$<CXX_COMPILER_VERSION>,19>>:BOOST_MPL_CFG_GCC='((__GNUC__ << 8) | __GNUC_MINOR__)'>" )

    # force boost to use decltype() for boost::result_of, required to compile without errors (ICC 17+18)
    target_compile_definitions(${target} INTERFACE
        $<$<AND:$<CXX_COMPILER_ID:Intel>,$<VERSION_LESS:$<CXX_COMPILER_VERSION>,19>>:BOOST_RESULT_OF_USE_DECLTYPE>)
endfunction(_workaround_icc)

