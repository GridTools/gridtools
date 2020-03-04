
# POST CONDITION: dst variable is set to one of HIPCC-AMDGPU/NVCC-CUDA/Clang-CUDA/NOTFOUND
function(detect_cuda_type dst prefer_clang)
    get_filename_component(cxx_name ${CMAKE_CXX_COMPILER} NAME)
    if(cxx_name STREQUAL hipcc)
        set(${dst} HIPCC-AMDGPU PARENT_SCOPE)
        return()
    endif()
    include(CheckLanguage)
    check_language(CUDA)
    find_package(CUDA QUIET)
    if (CMAKE_CUDA_COMPILER AND NOT (CMAKE_CXX_COMPILER_ID STREQUAL Clang AND prefer_clang))
        enable_language(CUDA)
    endif()

    if (CMAKE_CUDA_COMPILER AND
            NOT (CMAKE_CXX_COMPILER_ID STREQUAL Clang AND prefer_clang) AND
            CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 9.0)
        set(${dst} NVCC-CUDA PARENT_SCOPE)
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL Clang AND (CMAKE_CUDA_COMPILER OR CUDA_FOUND))
        set(${dst} Clang-CUDA PARENT_SCOPE)
    else()
        set(${dst} NOTFOUND PARENT_SCOPE)
    endif()
endfunction()
