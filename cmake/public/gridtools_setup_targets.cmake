# - If _config_mode == TRUE, prefix all targets with the GridTools:: namespace
# - for includes from this file, always use absolute filenames using _gt_gridtools_setup_targets_dir
#   as we use it in GridToolsConfig.cmake, too.
set(_gt_gridtools_setup_targets_dir ${CMAKE_CURRENT_LIST_DIR})
# TODO prefix all internal variables with _gt_
# TODO make aliases for all libraries in non config mode
macro(gridtools_setup_targets _config_mode clang_cuda_mode)
    include(${_gt_gridtools_setup_targets_dir}/detect_features.cmake)
    detect_cuda_type(GT_CUDA_TYPE "${clang_cuda_mode}")

    if(${_config_mode})
        set(_gt_namespace "GridTools::")
        set(_gt_imported IMPORTED)
    else()
        if((GT_CUDA_TYPE STREQUAL NVCC-CUDA) AND (CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME))
            # Do not enable the language if we are included from a super-project.
            # It is up to the super-project to enable CUDA.
            enable_language(CUDA)
        endif()
    endif()

    include(${_gt_gridtools_setup_targets_dir}/gridtools_helpers.cmake)

    # Add the gridtools_nvcc proxy if the CUDA language is enabled
    if (GT_CUDA_TYPE STREQUAL NVCC-CUDA)
        get_property(languages GLOBAL PROPERTY ENABLED_LANGUAGES)
        if("CUDA" IN_LIST languages)
            add_library(${_gt_namespace}gridtools_nvcc INTERFACE ${_gt_imported})
            # allow to call constexpr __host__ from constexpr __device__, e.g. call std::max in constexpr context
            target_compile_options(${_gt_namespace}gridtools_nvcc INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
        else()
            message(WARNING "CUDA is available, but was not enabled. If you want CUDA support, please enable the language in your project")
            set(GT_CUDA_TYPE NOTFOUND)
        endif()
    endif()

    find_package(MPI COMPONENTS CXX)
    include(${_gt_gridtools_setup_targets_dir}/workaround_mpi.cmake)
    _fix_mpi_flags()

    if(NOT (CMAKE_CXX_COMPILER_ID STREQUAL AppleClang AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 11.0))
        find_package(OpenMP COMPONENTS CXX)
    endif()

    if (GT_CUDA_TYPE STREQUAL NVCC-CUDA)
        set(GT_CUDA_ARCH_FLAG -arch)
    elseif (GT_CUDA_TYPE STREQUAL Clang-CUDA)
        set(GT_CUDA_ARCH_FLAG --cuda-gpu-arch)
    elseif (GT_CUDA_TYPE STREQUAL HIPCC-AMDGPU)
        set(GT_CUDA_ARCH_FLAG --amdgpu-target)
    endif()

    function(gridtools_cuda_setup type)
        if (type STREQUAL NVCC-CUDA)
            target_link_libraries(${_gt_namespace}gridtools_cuda INTERFACE ${_gt_namespace}gridtools_nvcc)
            find_library(cudart cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
            set(GT_CUDA_LIBRARIES ${cudart} PARENT_SCOPE)
            set(GT_CUDA_INCLUDE_DIRS ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} PARENT_SCOPE)
            message(STATUS "GridTools GPU mode: ${GT_CUDA_TYPE}, CUDA with NVCC")
        elseif(type STREQUAL Clang-CUDA)
            set(root_dir ${CUDAToolkit_BIN_DIR}/..)
            set(GT_CUDA_LIBRARIES ${CUDA_LIBRARIES} PARENT_SCOPE)
            set(GT_CUDA_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS} PARENT_SCOPE)
            target_compile_options(${_gt_namespace}gridtools_cuda INTERFACE -xcuda --cuda-path=${root_dir})
            target_link_libraries(${_gt_namespace}gridtools_cuda INTERFACE CUDA::cudart)
            message(STATUS "GridTools GPU mode: ${GT_CUDA_TYPE}, CUDA with clang")
        elseif(type STREQUAL HIPCC-AMDGPU)
            target_compile_options(${_gt_namespace}gridtools_cuda INTERFACE -xhip)
            message(STATUS "GridTools GPU mode: ${GT_CUDA_TYPE}, HIPCC on clang")
        endif()
    endfunction()

    if (GT_CUDA_TYPE)
        add_library(${_gt_namespace}gridtools_cuda INTERFACE ${_gt_imported})
        gridtools_cuda_setup(${GT_CUDA_TYPE})
    else()
        message(STATUS "GridTools GPU mode: ${GT_CUDA_TYPE}, no GPU support enabled.")
    endif()

    add_library(${_gt_namespace}storage_x86 INTERFACE ${_gt_imported})
    target_link_libraries(${_gt_namespace}storage_x86 INTERFACE ${_gt_namespace}gridtools)

    add_library(${_gt_namespace}storage_mc INTERFACE ${_gt_imported})
    target_link_libraries(${_gt_namespace}storage_mc INTERFACE ${_gt_namespace}gridtools)

    add_library(${_gt_namespace}backend_naive INTERFACE ${_gt_imported})
    target_link_libraries(${_gt_namespace}backend_naive INTERFACE ${_gt_namespace}gridtools)

    set(GT_BACKENDS naive)
    set(GT_ICO_BACKENDS naive)
    set(GT_STORAGES x86 mc)
    set(GT_GCL_ARCHS)

    if (TARGET ${_gt_namespace}gridtools_cuda)
        add_library(${_gt_namespace}backend_cuda INTERFACE ${_gt_imported})
        target_link_libraries(${_gt_namespace}backend_cuda INTERFACE ${_gt_namespace}gridtools ${_gt_namespace}gridtools_cuda)
        list(APPEND GT_BACKENDS cuda)
        list(APPEND GT_ICO_BACKENDS cuda)

        add_library(${_gt_namespace}storage_cuda INTERFACE ${_gt_imported})
        target_link_libraries(${_gt_namespace}storage_cuda INTERFACE ${_gt_namespace}gridtools ${_gt_namespace}gridtools_cuda)
        list(APPEND GT_STORAGES cuda)

        if(MPI_CXX_FOUND)
            add_library(${_gt_namespace}gcl_gpu INTERFACE ${_gt_imported})
            target_link_libraries(${_gt_namespace}gcl_gpu INTERFACE ${_gt_namespace}gridtools ${_gt_namespace}gridtools_cuda MPI::MPI_CXX)
        endif()

        add_library(${_gt_namespace}bc_gpu INTERFACE ${_gt_imported})
        target_link_libraries(${_gt_namespace}bc_gpu INTERFACE ${_gt_namespace}gridtools ${_gt_namespace}gridtools_cuda)

        add_library(${_gt_namespace}layout_transformation_gpu INTERFACE ${_gt_imported})
        target_link_libraries(${_gt_namespace}layout_transformation_gpu INTERFACE ${_gt_namespace}gridtools ${_gt_namespace}gridtools_cuda)

        list(APPEND GT_GCL_ARCHS gpu)
    endif()

    if (OpenMP_CXX_FOUND)
        add_library(${_gt_namespace}backend_x86 INTERFACE ${_gt_imported})
        target_link_libraries(${_gt_namespace}backend_x86 INTERFACE ${_gt_namespace}gridtools OpenMP::OpenMP_CXX)

        add_library(${_gt_namespace}backend_mc INTERFACE ${_gt_imported})
        target_link_libraries(${_gt_namespace}backend_mc INTERFACE ${_gt_namespace}gridtools OpenMP::OpenMP_CXX)

        if(MPI_CXX_FOUND)
            add_library(${_gt_namespace}gcl_cpu INTERFACE ${_gt_imported})
            target_link_libraries(${_gt_namespace}gcl_cpu INTERFACE ${_gt_namespace}gridtools OpenMP::OpenMP_CXX MPI::MPI_CXX)
        endif()
        add_library(${_gt_namespace}bc_cpu INTERFACE ${_gt_imported})
        target_link_libraries(${_gt_namespace}bc_cpu INTERFACE ${_gt_namespace}gridtools OpenMP::OpenMP_CXX)

        add_library(${_gt_namespace}layout_transformation_cpu INTERFACE ${_gt_imported})
        target_link_libraries(${_gt_namespace}layout_transformation_cpu INTERFACE ${_gt_namespace}gridtools OpenMP::OpenMP_CXX)

        list(APPEND GT_GCL_ARCHS cpu)

        list(APPEND GT_BACKENDS x86 mc)
        list(APPEND GT_ICO_BACKENDS x86)
    endif()
endmacro()
