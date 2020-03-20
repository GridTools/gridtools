set(CMAKE_EXPORT_NO_PACKAGE_REGISTRY ON CACHE BOOL "")
mark_as_advanced(CMAKE_EXPORT_NO_PACKAGE_REGISTRY)

option(GT_ENABLE_PYUTILS "If on, Python utility scripts will be configured" OFF)

# GT_CLANG_CUDA_MODE: decide if Clang-CUDA or NVCC
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(GT_CLANG_CUDA_MODE "AUTO" CACHE STRING
        "AUTO, Clang-CUDA or NVCC-CUDA; \
        AUTO = Use NVCC if language CUDA is enabled, else prefer Clang-CUDA.")
    set_property(CACHE GT_CLANG_CUDA_MODE PROPERTY STRINGS "AUTO;Clang-CUDA;NVCC-CUDA")
endif()

#if we are pointing to the default install path (usually system) we will disable installation of examples by default
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(_default_GT_INSTALL_EXAMPLES OFF)
else()
    set(_default_GT_INSTALL_EXAMPLES ON)
endif()
option(GT_INSTALL_EXAMPLES "Install example sources" ${_default_GT_INSTALL_EXAMPLES})
if(GT_INSTALL_EXAMPLES)
    set(GT_INSTALL_EXAMPLES_PATH "${CMAKE_INSTALL_PREFIX}/gridtools_examples" CACHE FILEPATH
        "Specifies where the source codes of examples should be installed")
    mark_as_advanced(CLEAR GT_INSTALL_EXAMPLES_PATH)
else()
    if(GT_INSTALL_EXAMPLES_PATH)
        mark_as_advanced(FORCE GT_INSTALL_EXAMPLES_PATH)
    endif()
endif()
