# try_clang_cuda()
# Parameters:
#    - result: result variable is set to Clang-CUDA or NOTFOUND
function(try_clang_cuda result)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        include(try_compile_clang_cuda)
        try_compile_clang_cuda("sm_60")
        if(GT_CLANG_CUDA_WORKS)
            set(${result} Clang-CUDA PARENT_SCOPE)
        endif()
    endif()
    set(${result} NOTFOUND PARENT_SCOPE)
endfunction()

# try_nvcc_cuda()
# Parameters:
#    - result: result variable is set to NVCC-CUDA  or NOTFOUND
function(try_nvcc_cuda result)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        set(${result} NVCC-CUDA PARENT_SCOPE)
    endif()
    set(${result} NOTFOUND PARENT_SCOPE)
endfunction()

# detect_cuda_type()
# Parameters:
#    - cuda_type: result variable is set to one of HIPCC-AMDGPU/NVCC-CUDA/Clang-CUDA/NOTFOUND
#    - clang_mode: AUTO, Clang-CUDA, NVCC-CUDA
#       - AUTO: Prefer NVCC-CUDA if the CUDA language is enabled, else try Clang-CUDA
#       - Clang-CUDA: Try Clang-CUDA or fail.
#       - NVCC-CUDA: Try NVCC-CUDA or fail.
function(detect_cuda_type cuda_type clang_mode)
    get_filename_component(cxx_name ${CMAKE_CXX_COMPILER} NAME)
    if(cxx_name STREQUAL "hipcc")
        include(try_compile_hip)
        try_compile_hip() #TODO use cache variable to avoid compiling each cmake run
        if(GT_HIP_WORKS)
            set(${cuda_type} HIPCC-AMDGPU PARENT_SCOPE)
            return()
        else()
            message(FATAL_ERROR "${cxx_name} wasn't able to compile a simple HIP program.")
        endif()
    endif()

    if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        try_nvcc_cuda(result)
        set(${cuda_type} ${result} PARENT_SCOPE)
        return()
    else() # Clang
        if(clang_mode STREQUAL "Clang-CUDA")
            try_clang_cuda(result)
            if(result)
                set(${cuda_type} ${result} PARENT_SCOPE)
                return()
            else()
                message(FATAL_ERROR "Clang-CUDA mode was selected, but doesn't work.")
            endif()
        elseif(clang_mode STREQUAL "NVCC-CUDA")
            try_nvcc_cuda(result)
            if(result)
                set(${cuda_type} ${result} PARENT_SCOPE)
                return()
            else()
                message(FATAL_ERROR "NVCC-CUDA mode was selected, but doesn't work.")
            endif()
        else() # AUTO
            get_property(languages GLOBAL PROPERTY ENABLED_LANGUAGES)
            if("CUDA" IN_LIST languages) # CUDA language is already enabled, prefer it
                set(${cuda_type} NVCC-CUDA PARENT_SCOPE)
                return()
            else()
                # Prefer Clang-CUDA
                try_clang_cuda(result)
                if(result)
                    set(${cuda_type} ${result} PARENT_SCOPE)
                    return()
                endif()

                # Clang-CUDA doesn't work, try NVCC
                try_nvcc_cuda(result)
                if(result)
                    set(${cuda_type} ${result} PARENT_SCOPE)
                    return()
                endif()

                set(${cuda_type} NOTFOUND PARENT_SCOPE)
                return()
            endif()
        endif()
    endif()
    set(${cuda_type} NOTFOUND PARENT_SCOPE)
endfunction()
