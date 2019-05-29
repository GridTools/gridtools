cmake_minimum_required(VERSION 3.12)
project(computation LANGUAGES CXX)

# enable CUDA if it is found on the system
# include(../workaround_check_language.cmake) # see https://gitlab.kitware.com/cmake/cmake/issues/19013
include(CheckLanguage) # see https://gitlab.kitware.com/cmake/cmake/issues/19013
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    # 2) Enable the CUDA language if you want to run your code on a CUDA-capable GPU. This
    #    must be done before calling `find_package(GridTools)`
    enable_language(CUDA)
endif()

find_package(GridTools 1.1.0 HINTS ../../install/lib/cmake)

add_executable(driver_mc driver.cpp interpolate_stencil.cpp)
target_link_libraries(driver_mc GridTools::gridtools)

if (CMAKE_CUDA_COMPILER)
    if(GRIDTOOLS_HAS_BACKEND_CUDA)
        add_library(stencil_lib_cuda interpolate_stencil.cu)
        target_compile_definitions(stencil_lib_cuda PUBLIC USE_GPU)
        target_link_libraries(stencil_lib_cuda PUBLIC GridTools::gridtools)

        add_executable(driver_cuda driver.cpp)
        target_link_libraries(driver_cuda PUBLIC stencil_lib_cuda)
    endif()
endif()
