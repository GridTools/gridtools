# If install_mode == TRUE, prefix all targets with the GridTools:: namespace
macro(gridtools_setup_targets install_mode)
    if(NOT DEFINED GT_CUDA_TYPE)
        # GT_CUDA_TYPE needs to be run before
        # (but cannot be included here as in non-GridToolsConfig mode we need to enable a language)
        # TODO consider passing the information via parameter
        message(FATAL_ERROR "CMake configuration issue. This is a bug in GridTools, please open an issue.")
    endif()

    if(install_mode)
        set(_gt_namespace "GridTools::")
    endif()

    include(gridtools_helpers) #TODO consider inlining the functions in this file

    # Add the gridtools_nvcc proxy if the CUDA language is enabled
    if (GT_CUDA_TYPE STREQUAL NVCC-CUDA)
        get_property(languages GLOBAL PROPERTY ENABLED_LANGUAGES)
        if("CUDA" IN_LIST languages)
            add_library(${_gt_namespace}gridtools_nvcc INTERFACE)
            # allow to call constexpr __host__ from constexpr __device__, e.g. call std::max in constexpr context
            target_compile_options(${_gt_namespace}gridtools_nvcc INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
        else()
            message(WARNING "CUDA is available, but was not enabled. If you want CUDA support, please enable the language in your project")
            set(GT_CUDA_TYPE NOTFOUND)
        endif()
    endif()

    find_package(MPI COMPONENTS CXX)
    include(workaround_mpi)
    _fix_mpi_flags()

    if (GT_CUDA_TYPE STREQUAL NVCC-CUDA)
        set(GT_CUDA_ARCH_FLAG -arch)
    elseif (GT_CUDA_TYPE STREQUAL Clang-CUDA)
        set(GT_CUDA_ARCH_FLAG --cuda-gpu-arch)
    elseif (GT_CUDA_TYPE STREQUAL HIPCC-AMDGPU)
        set(GT_CUDA_ARCH_FLAG --amdgpu-target)
    endif()

    function(gridtools_cuda_setup type)
        if (type STREQUAL NVCC-CUDA)
            target_link_libraries(${_gt_namespace}gridtools_cuda INTERFACE gridtools_nvcc)
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
        add_library(${_gt_namespace}gridtools_cuda INTERFACE)
        gridtools_cuda_setup(${GT_CUDA_TYPE})
    else()
        message(STATUS "GridTools GPU mode: ${GT_CUDA_TYPE}, no GPU support enabled.")
    endif()

    add_library(${_gt_namespace}gridtools INTERFACE)
    target_compile_features(${_gt_namespace}gridtools INTERFACE cxx_std_14)
    target_link_libraries(${_gt_namespace}gridtools INTERFACE Boost::boost)
    target_compile_definitions(${_gt_namespace}gridtools INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:BOOST_PP_VARIADICS=1>)
    target_include_directories(${_gt_namespace}gridtools
            INTERFACE
            $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/>
            $<INSTALL_INTERFACE:include>
            )

    add_library(${_gt_namespace}storage_x86 INTERFACE)
    target_link_libraries(${_gt_namespace}storage_x86 INTERFACE gridtools)

    add_library(${_gt_namespace}storage_mc INTERFACE)
    target_link_libraries(${_gt_namespace}storage_mc INTERFACE gridtools)

    add_library(${_gt_namespace}backend_naive INTERFACE)
    target_link_libraries(${_gt_namespace}backend_naive INTERFACE gridtools)

    set(GT_BACKENDS naive)
    set(GT_ICO_BACKENDS naive)
    set(GT_STORAGES x86 mc)
    set(GT_GCL_ARCHS)

    if (TARGET ${_gt_namespace}gridtools_cuda)
        add_library(${_gt_namespace}backend_cuda INTERFACE)
        target_link_libraries(${_gt_namespace}backend_cuda INTERFACE gridtools gridtools_cuda)
        list(APPEND GT_BACKENDS cuda)
        list(APPEND GT_ICO_BACKENDS cuda)

        add_library(${_gt_namespace}storage_cuda INTERFACE)
        target_link_libraries(${_gt_namespace}storage_cuda INTERFACE gridtools gridtools_cuda)
        list(APPEND GT_STORAGES cuda)

        if(MPI_CXX_FOUND)
            add_library(${_gt_namespace}gcl_gpu INTERFACE)
            target_link_libraries(${_gt_namespace}gcl_gpu INTERFACE gridtools gridtools_cuda MPI::MPI_CXX)
        endif()

        add_library(${_gt_namespace}bc_gpu INTERFACE)
        target_link_libraries(${_gt_namespace}bc_gpu INTERFACE gridtools gridtools_cuda)

        add_library(${_gt_namespace}layout_transformation_gpu INTERFACE)
        target_link_libraries(${_gt_namespace}layout_transformation_gpu INTERFACE gridtools gridtools_cuda)

        list(APPEND GT_GCL_ARCHS gpu)
    endif()

    if (OpenMP_CXX_FOUND)
        add_library(${_gt_namespace}backend_x86 INTERFACE)
        target_link_libraries(${_gt_namespace}backend_x86 INTERFACE gridtools OpenMP::OpenMP_CXX)

        add_library(${_gt_namespace}backend_mc INTERFACE)
        target_link_libraries(${_gt_namespace}backend_mc INTERFACE gridtools OpenMP::OpenMP_CXX)

        if(MPI_CXX_FOUND)
            add_library(${_gt_namespace}gcl_cpu INTERFACE)
            target_link_libraries(${_gt_namespace}gcl_cpu INTERFACE gridtools OpenMP::OpenMP_CXX MPI::MPI_CXX)
        endif()
        add_library(${_gt_namespace}bc_cpu INTERFACE)
        target_link_libraries(${_gt_namespace}bc_cpu INTERFACE gridtools OpenMP::OpenMP_CXX)

        add_library(${_gt_namespace}layout_transformation_cpu INTERFACE)
        target_link_libraries(${_gt_namespace}layout_transformation_cpu INTERFACE gridtools OpenMP::OpenMP_CXX)

        list(APPEND GT_GCL_ARCHS cpu)

        list(APPEND GT_BACKENDS x86 mc)
        list(APPEND GT_ICO_BACKENDS x86)
    endif()
endmacro()
