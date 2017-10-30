

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
      message( SEND_ERROR "
        Could not deduce CUDA_ARCH from GPU_DEVICE=${GPU_DEVICE}.
        Possible options: K40, K80, P100.
        Or try to manually set CUDA_ARCH (e.g. for P100 : \"CUDA_ARCH=sm_60\")
        " )
    endif()

  endif()


  list( APPEND GRIDTOOLS_SYSTEM_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS} )
  list( APPEND GRIDTOOLS_LINK_LIBRARIES ${CUDA_CUDART_LIBRARY} )
  list( APPEND GRIDTOOLS_DEFINITIONS "-D_USE_GPU_" )
  
  macro( add_nvcc_flag flag )
    set( CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${flag}" )
  endmacro()

  string( REPLACE "." "" CUDA_VERSION_INT ${CUDA_VERSION} )
  add_nvcc_flag( "-DCUDA_VERSION=${CUDA_VERSION_INT}" )
  add_nvcc_flag( "-arch=${CUDA_ARCH}" )
  add_nvcc_flag( "--std=c++11" )

  if( ENABLE_WERROR )
    # Unfortunately we cannot treat all errors as warnings, we have to specify each warning.
    # The only supported warning in CUDA8 is cross-execution-space-call
    add_nvcc_flag( "--Werror cross-execution-space-call -Xptxas --warning-as-error --nvlink-options --warning-as-error" )
  endif()

  # Suppress nvcc warnings
  foreach( _diag 
              dupl_calling_convention code_is_unreachable
              implicit_return_from_non_void_function
              calling_convention_not_allowed
              conflicting_calling_conventions )
    add_nvcc_flag( "-Xcudafe --diag_suppress=${_diag}" )
  endforeach()

endif()

## clang ##
if( (CUDA_HOST_COMPILER MATCHES "(C|c?)lang") OR (CMAKE_CXX_COMPILER_ID MATCHES "(C|c?)lang") )
  list( APPEND GRIDTOOLS_CXX_FLAGS "-ftemplate-depth-1024" )
endif()
