
macro( gridtools_add_nvcc_flags )
  foreach( flag ${ARGV} )
    set( CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${flag} )
  endforeach()
  list( APPEND GRIDTOOLS_NVCC_FLAGS "${ARGV}" )
endmacro()
