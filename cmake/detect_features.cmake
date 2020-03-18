
# POST CONDITION: dst variable is set to one of HIPCC-AMDGPU/NVCC-CUDA/Clang-CUDA/NOTFOUND
function(detect_cuda_type dst prefer_clang)
    get_filename_component(cxx_name ${CMAKE_CXX_COMPILER} NAME)
    if(cxx_name STREQUAL "hipcc")
        include(try_compile_hip)
        try_compile_hip() #TODO use cache variable to avoid compiling each cmake run
        if(GT_HIP_WORKS)
            set(${dst} HIPCC-AMDGPU PARENT_SCOPE)
            return()
        else()
            message(FATAL_ERROR "${cxx_name} wasn't able to compile a simple HIP program.")
        endif()
    endif()

    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND prefer_clang)
        find_package(CUDAToolkit)
        if(CUDAToolkit_FOUND)
            include(try_compile_clang_cuda)
            try_compile_clang_cuda()
            if(GT_CLANG_CUDA_WORKS)
                set(${dst} Clang-CUDA PARENT_SCOPE)
                return()
            endif()
        endif()
        set(${dst} NOTFOUND PARENT_SCOPE)
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
