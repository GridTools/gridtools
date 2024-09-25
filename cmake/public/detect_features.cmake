# try_clang_cuda()
# Parameters:
#    - gt_result: result variable is set to Clang-CUDA or NOTFOUND
function(try_clang_cuda gt_result)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        include(try_compile_clang_cuda)
        try_compile_clang_cuda(GT_CLANG_CUDA_WORKS "sm_60")
        if(GT_CLANG_CUDA_WORKS)
            set(${gt_result} Clang-CUDA PARENT_SCOPE)
            return()
        endif()
    endif()
    set(${gt_result} NOTFOUND PARENT_SCOPE)
endfunction()

# try_nvcc_cuda()
# Parameters:
#    - gt_result: result variable is set to NVCC-CUDA or NOTFOUND
function(try_nvcc_cuda gt_result)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        set(${gt_result} NVCC-CUDA PARENT_SCOPE)
        return()
    endif()
    set(${gt_result} NOTFOUND PARENT_SCOPE)
endfunction()

function(try_hip gt_result)
    include(CheckLanguage)
    check_language(HIP)
    if(CMAKE_HIP_COMPILER)
        set(${gt_result} HIPCC-AMDGPU PARENT_SCOPE)
        return()
    endif()
    set(${gt_result} NOTFOUND PARENT_SCOPE)
endfunction()

# detect_cuda_type()
# Parameters:
#    - cuda_type: result variable is set to one of HIPCC-AMDGPU/NVCC-CUDA/Clang-CUDA/NOTFOUND
#    - mode: AUTO, HIP, Clang-CUDA, NVCC-CUDA
#       - AUTO: Prefer NVCC-CUDA if the CUDA language is enabled, prefer HIP if the HIP language is enabled, else try Clang-CUDA, else try HIP.
#       - HIP: Try HIP or fail.
#       - Clang-CUDA: Try Clang-CUDA or fail.
#       - NVCC-CUDA: Try NVCC-CUDA or fail.
function(detect_cuda_type cuda_type mode)
    if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        # not Clang, therefore the only option is NVCC
        try_nvcc_cuda(gt_result)
        set(${cuda_type} ${gt_result} PARENT_SCOPE)
        return()
    else() # Clang
        string(TOLOWER "${mode}" _lower_case_mode)
        if(_lower_case_mode STREQUAL "clang-cuda")
            try_clang_cuda(gt_result)
            if(gt_result)
                set(${cuda_type} ${gt_result} PARENT_SCOPE)
                return()
            else()
                message(FATAL_ERROR "Clang-CUDA mode was selected, but doesn't work.")
            endif()
        elseif(_lower_case_mode STREQUAL "nvcc-cuda")
            try_nvcc_cuda(gt_result)
            if(gt_result)
                set(${cuda_type} ${gt_result} PARENT_SCOPE)
                return()
            else()
                message(FATAL_ERROR "NVCC-CUDA mode was selected, but doesn't work.")
            endif()
        elseif(_lower_case_mode STREQUAL "hip")
            try_hip(gt_result)
            if(gt_result)
                set(${cuda_type} ${gt_result} PARENT_SCOPE)
                return()
            else()
                message(FATAL_ERROR "HIP mode was selected, but doesn't work.")
            endif()
        elseif(_lower_case_mode STREQUAL "auto") # AUTO
            get_property(languages GLOBAL PROPERTY ENABLED_LANGUAGES)
            if("CUDA" IN_LIST languages) # CUDA language is already enabled, prefer it
                set(${cuda_type} NVCC-CUDA PARENT_SCOPE)
                return()
            elseif("HIP" IN_LIST languages) # HIP language is already enabled, prefer it
                set(${cuda_type} HIPCC-AMDGPU PARENT_SCOPE)
                return()
            else()
                # Prefer Clang-CUDA
                try_clang_cuda(gt_result)
                if(gt_result)
                    set(${cuda_type} ${gt_result} PARENT_SCOPE)
                    return()
                endif()

                # Clang-CUDA doesn't work, try NVCC
                try_nvcc_cuda(gt_result)
                if(gt_result)
                    set(${cuda_type} ${gt_result} PARENT_SCOPE)
                    return()
                endif()

                # No CUDA variant works, try HIP
                try_hip(gt_result)
                if(gt_result)
                    set(${cuda_type} ${gt_result} PARENT_SCOPE)
                    return()
                endif()

                set(${cuda_type} NOTFOUND PARENT_SCOPE)
            endif()
        else()
            message(FATAL_ERROR "CUDA/HIP mode set to invalid value ${mode}")
        endif()
    endif()
    set(${cuda_type} NOTFOUND PARENT_SCOPE)
endfunction()
