
macro( gridtools_setup_CUDA )

  set( GRIDTOOLS_HAVE_CUDA 0 )

  if( ENABLE_CUDA )
    if( ENABLE_CUDA_REQUIRED )
      set( _find_cuda REQUIRED )
    else()
      set( _find_cuda QUIET )
    endif()

    find_package( CUDA 6.0 ${_find_cuda} )

    if( NOT CUDA_FOUND )

      message( STATUS "CUDA not found" )
      set( GRIDTOOLS_HAVE_CUDA 0 )

    else()

      message( STATUS "CUDA detected: " ${CUDA_VERSION} )
      set( GRIDTOOLS_HAVE_CUDA 1 )

      set( CUDA_PROPAGATE_HOST_FLAGS ON )
      set( CUDA_LIBRARIES "" ) # Is this required?

      set( CUDA_ARCH "CUDA_ARCH-NOTFOUND" CACHE STRING "CUDA architecture (e.g. sm_35, sm_37, sm_60); precedence over GPU_DEVICE" )

      if( NOT CUDA_ARCH )

        if    ( ${GPU_DEVICE} STREQUAL "P100" )
          set( CUDA_ARCH "sm_60" )

        elseif( ${GPU_DEVICE} STREQUAL "K80"  )
          set( CUDA_ARCH "sm_37" )

        elseif( ${GPU_DEVICE} STREQUAL "K40"  )
          set( CUDA_ARCH "sm_35" )

        else()

          if( ENABLE_CUDA_REQUIRED )
            set( _message FATAL_ERROR )
          else()
            set( GRIDTOOLS_HAVE_CUDA 0 )
            set( _message WARNING )
            set( _add_text "
      Ignoring error (set ENABLE_CUDA=OFF to silence warning).
      To force failure, set ENABLE_CUDA=REQUIRED
      " )
          endif()
          message( ${_message} "
            Could not deduce CUDA_ARCH from GPU_DEVICE=${GPU_DEVICE}.
            Possible options for GPU_DEVICE: K40, K80, P100.
            
            Alternatively:
              - Set CUDA_ARCH (e.g. for P100 : \"CUDA_ARCH=sm_60\")
              - Set ENABLE_CUDA=OFF
            ${_add_text}
            " )
        endif()

      endif()

      if( GRIDTOOLS_HAVE_CUDA )

        list( APPEND GRIDTOOLS_SYSTEM_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS} )
        list( APPEND GRIDTOOLS_LINK_LIBRARIES ${CUDA_CUDART_LIBRARY} )
        list( APPEND GRIDTOOLS_DEFINITIONS "-D_USE_GPU_" )
        

        string( REPLACE "." "" CUDA_VERSION_INT ${CUDA_VERSION} )
        gridtools_add_nvcc_flags( -DCUDA_VERSION=${CUDA_VERSION_INT} )
        gridtools_add_nvcc_flags( -arch=${CUDA_ARCH} )
        gridtools_add_nvcc_flags( --compiler-options -fPIC )

        if( ENABLE_WERROR )
          # Unfortunately we cannot treat all errors as warnings, we have to specify each warning.
          # The only supported warning in CUDA8 is cross-execution-space-call
          gridtools_add_nvcc_flags( --Werror cross-execution-space-call -Xptxas --warning-as-error --nvlink-options --warning-as-error )
        endif()

        # Suppress nvcc warnings
        foreach( _diag 
                    dupl_calling_convention code_is_unreachable
                    implicit_return_from_non_void_function
                    calling_convention_not_allowed
                    conflicting_calling_conventions )
          gridtools_add_nvcc_flags( -Xcudafe --diag_suppress=${_diag} )
        endforeach()

      endif()

    endif()

  endif()

endmacro()
