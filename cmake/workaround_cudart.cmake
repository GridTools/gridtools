# There is no find_package(CUDART) in CMake. So we need to do that manually.
# https://gitlab.kitware.com/cmake/cmake/issues/17816
macro(_fix_cudart_library)
  set(CUDA_TOOLKIT_ROOT_DIR "${CMAKE_CUDA_COMPILER}")
  find_library(CUDA_CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
  mark_as_advanced(CUDA_CUDART_LIBRARY)
  if (NOT CUDA_CUDART_LIBRARY)
      message(FATAL_ERROR "Cuda runtime was not found")
  endif()
endmacro()
