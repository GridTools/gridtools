string(TOLOWER "${CMAKE_BUILD_TYPE}" PYUTILS_BUILD_TYPE)

if(TARGET gridtools_cuda)
    if (TARGET gridtools_nvcc)
        set(PYUTILS_COMPILER "${CMAKE_CUDA_COMPILER} ${CMAKE_CUDA_COMPILER_VERSION} (${CMAKE_CXX_COMPILER} ${CMAKE_CXX_COMPILER_VERSION})")
    else()
        set(PYUTILS_COMPILER "${CMAKE_CXX_COMPILER} ${CMAKE_CXX_COMPILER_VERSION} (${CMAKE_CUDA_COMPILER} ${CMAKE_CUDA_COMPILER_VERSION})")
    endif()
else()
    set(PYUTILS_COMPILER "${CMAKE_CXX_COMPILER} ${CMAKE_CXX_COMPILER_VERSION}")
endif()

if(GT_SINGLE_PRECISION)
    set(PYUTILS_PRECISION "float")
else()
    set(PYUTILS_PRECISION "double")
endif()

if(TARGET backend_x86)
    list(APPEND PYUTILS_BACKEND "x86")
endif()

if(TARGET backend_naive)
    list(APPEND PYUTILS_BACKEND "naive")
endif()

if(TARGET backend_cuda)
    list(APPEND PYUTILS_BACKEND "cuda")
endif()

if(TARGET backend_mc)
    list(APPEND PYUTILS_BACKEND "mc")
endif()

file(COPY . DESTINATION . FILES_MATCHING PATTERN *.py)
configure_file(pyutils/buildinfo.py.in pyutils/buildinfo.py)
