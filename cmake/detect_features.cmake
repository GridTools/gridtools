
# POST CONDITION: dst variable is set to one of HIPCC-AMDGPU/NVCC-CUDA/Clang-CUDA/NOTFOUND
function(detect_cuda_type dst prefer_clang)
    get_filename_component(cxx_name ${CMAKE_CXX_COMPILER} NAME)
    if(cxx_name STREQUAL "hipcc")
        set(${dst} HIPCC-AMDGPU PARENT_SCOPE)
        return()
    endif()

    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND prefer_clang)
        find_package(CUDAToolkit)
        if(CUDAToolkit_FOUND)
            # TODO here we need to run a test if we can compile a simple test program
            set(${dst} Clang-CUDA PARENT_SCOPE)
        else()
            set(${dst} NOTFOUND PARENT_SCOPE)
        endif()
        return()
    endif()

    # either we are not clang or we are clang but don't prefer clang
    include(CheckLanguage)
    check_language(CUDA)
    if (CMAKE_CUDA_COMPILER)
        set(${dst} NVCC-CUDA PARENT_SCOPE)
        return()
    endif()

    set(${dst} NOTFOUND PARENT_SCOPE)
endfunction()
