macro( gridtools_cmake_workarounds )

  if( GRIDTOOLS_HAVE_CUDA )

    # The propagation of includes and defines to nvcc is broken for CMake version < 3.7
    # Note: this makes the install non-relocatable for affected versions
    if( CMAKE_VERSION VERSION_LESS 3.7.0 )
       foreach( def ${GRIDTOOLS_DEFINITIONS} )
          gridtools_add_nvcc_flags( ${def} )
       endforeach()
       foreach( dir ${GRIDTOOLS_SOURCE_DIR}/include
                    ${GRIDTOOLS_SOURCE_DIR}/include/gridtools
                    ${CMAKE_INSTALL_PREFIX}/include
                    ${CMAKE_INSTALL_PREFIX}/include/gridtools )
          gridtools_add_nvcc_flags( -I ${dir} )
      endforeach()
    endif()

    # The propagation of SYSTEM include dirs to nvcc is STILL broken (tested up to CMake version 3.10)
    # Carlos Osuna reported this upstream
    #    https://cmake.org/pipermail/cmake/2016-April/063234.html 
    # This is a workaround
    foreach( dir ${GRIDTOOLS_SYSTEM_INCLUDE_DIRS} )
      gridtools_add_nvcc_flags( -isystem ${dir} )
    endforeach()

  endif()

endmacro()