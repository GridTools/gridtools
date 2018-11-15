# There is no find_package(CUDART) in CMake. So we need to do that manually.
# https://gitlab.kitware.com/cmake/cmake/issues/17816
macro(_fix_cudart_library)
  set(CUDA_TOOLKIT_ROOT_DIR "${CMAKE_CUDA_COMPILER}")
  get_filename_component(CUDA_TOOLKIT_ROOT_DIR "${CUDA_TOOLKIT_ROOT_DIR}" DIRECTORY)
  get_filename_component(CUDA_TOOLKIT_ROOT_DIR "${CUDA_TOOLKIT_ROOT_DIR}" DIRECTORY)
  find_library(CUDA_CUDART_LIBRARY cudart
               HINTS
               "${CUDA_TOOLKIT_ROOT_DIR}/lib64"
               "${CUDA_TOOLKIT_ROOT_DIR}/lib"
               "${CUDA_TOOLKIT_ROOT_DIR}"
               )
  mark_as_advanced(CUDA_CUDART_LIBRARY)
  if (NOT CUDA_CUDART_LIBRARY)
      message(FATAL_ERROR "Cuda Runtime was not found")
  endif()
endmacro()
