macro( gridtools_setup_cxx )

    # CMake versions < 3.8 don't support the language compile feature cxx_std_11.
    # However only CMake version 3.9 onwards supports Cray implementation.

    if(${CMAKE_VERSION} VERSION_LESS "3.9.0") 
      if( CMAKE_CXX_COMPILER_ID MATCHES Cray )
        set( CXX11_flag "-hstd=c++11" )
      else()
        set( CXX11_flag "-std=c++11" )
      endif()
      message( STATUS "C++11 flag: ${CXX11_flag}  ( Consider CMake v3.9.0 to autodetect C++11 flag )" )
      list( APPEND GRIDTOOLS_CXX_FLAGS ${CXX11_flag} )
    else()
      list( APPEND GRIDTOOLS_COMPILE_FEATURES "cxx_std_11" )
    endif()

    ## clang ##
    if( (CUDA_HOST_COMPILER MATCHES "(C|c?)lang") OR (CMAKE_CXX_COMPILER_ID MATCHES "(C|c?)lang") )
      list( APPEND GRIDTOOLS_CXX_FLAGS "-ftemplate-depth-1024" )
    endif()

endmacro()