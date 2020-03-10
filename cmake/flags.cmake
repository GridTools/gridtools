
set(CMAKE_EXPORT_NO_PACKAGE_REGISTRY ON CACHE BOOL "")
mark_as_advanced(CMAKE_EXPORT_NO_PACKAGE_REGISTRY)

option(GT_ENABLE_PYUTILS "If on, Python utility scripts will be configured" OFF)

option(GT_PREFER_CLANG_CUDA_OVER_NVCC_CUDA "if ON, CUDA code will be compiled with clang" ON)

option(GT_ENABLE_EXPERIMENTAL_REPOSITORY "Enables downloading the gridtools_experimental repository" OFF )

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
