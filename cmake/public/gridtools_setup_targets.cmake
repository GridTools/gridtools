set(_gt_gridtools_setup_targets_dir ${CMAKE_CURRENT_LIST_DIR})
set(GT_NAMESPACE "GridTools::")

# _gt_add_library()
# In config mode (called from GridToolsConfig.cmake), we create an IMPORTED target with namespace
# In normal mode (called from main CMakeLists.txt), we create a target without namespace and an alias
# This function should only be used to create INTERFACE targets in the context of gridtools_setup_targets().
function(_gt_add_library _config_mode name)
    if(${_config_mode})
        add_library(${GT_NAMESPACE}${name} INTERFACE IMPORTED)
    else()
        add_library(${name} INTERFACE)
        add_library(${GT_NAMESPACE}${name} ALIAS ${name})
    endif()
    list(APPEND GT_AVAILABLE_TARGETS ${GT_NAMESPACE}${name})
    set(GT_AVAILABLE_TARGETS ${GT_AVAILABLE_TARGETS} PARENT_SCOPE)
endfunction()

# gridtools_setup_targets()
# This macro is used in the main CMakeLists.txt (_config_mode == FALSE) and for setting up the targets in the
# GridToolsConfig.cmake, i.e. from a GridTools installation (_config_mode == TRUE).
#
# For this reason some restrictions apply to commands in this macro:
# - All targets created in this macro need to be created with _gt_add_library() which will create proper namespace
#   prefix and aliases.
# - Within this macro, all references to targets created with _gt_add_library() need to be prefixed with
#   ${_gt_namespace}, e.g. target_link_libraries(${_gt_namespace}my_tgt INTERFACE ${_gt_namespace}_my_other_tgt).
# - Including other CMake files should be done with absolute paths. Use ${_gt_gridtools_setup_targets_dir} to refer
#   to the directory where this files lives.
macro(_gt_setup_targets _config_mode clang_cuda_mode)
    set(GT_AVAILABLE_TARGETS)

    include(${_gt_gridtools_setup_targets_dir}/detect_features.cmake)
    detect_cuda_type(GT_CUDA_TYPE "${clang_cuda_mode}")

    if(${_config_mode})
        set(_gt_namespace ${GT_NAMESPACE})
        set(_gt_imported "IMPORTED")
    else()
        if((GT_CUDA_TYPE STREQUAL NVCC-CUDA) AND (CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME))
            # Do not enable the language if we are included from a super-project.
            # It is up to the super-project to enable CUDA.
            enable_language(CUDA)
        endif()
    endif()

    include(${_gt_gridtools_setup_targets_dir}/gridtools_helpers.cmake)

    # Add the _gridtools_nvcc proxy if the CUDA language is enabled
    if (GT_CUDA_TYPE STREQUAL NVCC-CUDA)
        get_property(languages GLOBAL PROPERTY ENABLED_LANGUAGES)
        if("CUDA" IN_LIST languages)
            add_library(_gridtools_nvcc INTERFACE ${_gt_imported})
            # allow to call constexpr __host__ from constexpr __device__, e.g. call std::max in constexpr context
            target_compile_options(_gridtools_nvcc INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
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
            target_link_libraries(_gridtools_cuda INTERFACE _gridtools_nvcc)
            find_library(cudart cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
            set(GT_CUDA_LIBRARIES ${cudart} PARENT_SCOPE)
            set(GT_CUDA_INCLUDE_DIRS ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} PARENT_SCOPE)
        elseif(type STREQUAL Clang-CUDA)
            set(_gt_setup_root_dir ${CUDAToolkit_BIN_DIR}/..)
            set(GT_CUDA_LIBRARIES ${CUDA_LIBRARIES} PARENT_SCOPE)
            set(GT_CUDA_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS} PARENT_SCOPE)
            target_compile_options(_gridtools_cuda INTERFACE -xcuda --cuda-path=${_gt_setup_root_dir})
            target_link_libraries(_gridtools_cuda INTERFACE CUDA::cudart)
        elseif(type STREQUAL HIPCC-AMDGPU)
            target_compile_options(_gridtools_cuda INTERFACE -xhip)
        endif()
    endfunction()

    if (GT_CUDA_TYPE)
        add_library(_gridtools_cuda INTERFACE ${_gt_imported})
        gridtools_cuda_setup(${GT_CUDA_TYPE})
    endif()

    _gt_add_library(${_config_mode} storage_x86)
    target_link_libraries(${_gt_namespace}storage_x86 INTERFACE ${_gt_namespace}gridtools)

    _gt_add_library(${_config_mode} storage_mc)
    target_link_libraries(${_gt_namespace}storage_mc INTERFACE ${_gt_namespace}gridtools)

    _gt_add_library(${_config_mode} backend_naive)
    target_link_libraries(${_gt_namespace}backend_naive INTERFACE ${_gt_namespace}gridtools)

    set(GT_BACKENDS naive) #TODO move outside of this file
    set(GT_ICO_BACKENDS naive)
    set(GT_STORAGES x86 mc)
    set(GT_GCL_ARCHS)

    if (TARGET _gridtools_cuda)
        _gt_add_library(${_config_mode} backend_cuda)
        target_link_libraries(${_gt_namespace}backend_cuda INTERFACE ${_gt_namespace}gridtools _gridtools_cuda)
        list(APPEND GT_BACKENDS cuda)
        list(APPEND GT_ICO_BACKENDS cuda)

        _gt_add_library(${_config_mode} storage_cuda)
        target_link_libraries(${_gt_namespace}storage_cuda INTERFACE ${_gt_namespace}gridtools _gridtools_cuda)
        list(APPEND GT_STORAGES cuda)

        if(MPI_CXX_FOUND)
            _gt_add_library(${_config_mode} gcl_gpu)
            target_link_libraries(${_gt_namespace}gcl_gpu INTERFACE ${_gt_namespace}gridtools _gridtools_cuda MPI::MPI_CXX)
        endif()

        _gt_add_library(${_config_mode} bc_gpu)
        target_link_libraries(${_gt_namespace}bc_gpu INTERFACE ${_gt_namespace}gridtools _gridtools_cuda)

        _gt_add_library(${_config_mode} layout_transformation_gpu)
        target_link_libraries(${_gt_namespace}layout_transformation_gpu INTERFACE ${_gt_namespace}gridtools _gridtools_cuda)

        list(APPEND GT_GCL_ARCHS gpu)
    endif()

    if (OpenMP_CXX_FOUND)
        _gt_add_library(${_config_mode} backend_x86)
        target_link_libraries(${_gt_namespace}backend_x86 INTERFACE ${_gt_namespace}gridtools OpenMP::OpenMP_CXX)

        _gt_add_library(${_config_mode} backend_mc)
        target_link_libraries(${_gt_namespace}backend_mc INTERFACE ${_gt_namespace}gridtools OpenMP::OpenMP_CXX)

        if(MPI_CXX_FOUND)
            _gt_add_library(${_config_mode} gcl_cpu)
            target_link_libraries(${_gt_namespace}gcl_cpu INTERFACE ${_gt_namespace}gridtools OpenMP::OpenMP_CXX MPI::MPI_CXX)
        endif()
        _gt_add_library(${_config_mode} bc_cpu)
        target_link_libraries(${_gt_namespace}bc_cpu INTERFACE ${_gt_namespace}gridtools OpenMP::OpenMP_CXX)

        _gt_add_library(${_config_mode} layout_transformation_cpu)
        target_link_libraries(${_gt_namespace}layout_transformation_cpu INTERFACE ${_gt_namespace}gridtools OpenMP::OpenMP_CXX)

        list(APPEND GT_GCL_ARCHS cpu)

        list(APPEND GT_BACKENDS x86 mc)
        list(APPEND GT_ICO_BACKENDS x86)
    endif()
endmacro()

function(_gt_print_configuration_summary)
    message(STATUS "GridTools configuration summary")
    message(STATUS "  Available targets: ${GT_AVAILABLE_TARGETS}")
    message(STATUS "  GPU mode: ${GT_CUDA_TYPE}")
endfunction()
