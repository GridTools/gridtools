set(_gt_gridtools_setup_targets_dir ${CMAKE_CURRENT_LIST_DIR})
set(GT_NAMESPACE "GridTools::")

# _gt_depends_on()
# Note: This function relies on the unique name of a dependency. If libraries get a namespace prefix in exported
# context, this function may fail. Therefore, use of this function with care!
function(_gt_depends_on dst lib dependency)
    if (NOT TARGET ${dependency})
        message(AUTHOR_WARNING "_gt_depends_on() argument ${dependency} is not a target.")
    elseif (lib STREQUAL dependency)
        set(${dst} ON PARENT_SCOPE)
	return()
    elseif (TARGET ${lib})
        get_target_property(tgt_type ${lib} TYPE)
        set(deps)
        if(NOT tgt_type STREQUAL "INTERFACE_LIBRARY")
            get_target_property(deps ${lib} LINK_LIBRARIES)
        endif()
        get_target_property(deps_interface ${lib} INTERFACE_LINK_LIBRARIES)
        if (deps OR deps_interface)
            foreach(dep IN LISTS deps deps_interface)
                _gt_depends_on(child ${dep} ${dependency})
                if (child)
                    set(${dst} ON PARENT_SCOPE)
                    return()
                endif()
            endforeach()
        endif()
    endif()
    set(${dst} OFF PARENT_SCOPE)
endfunction()

function(_gt_depends_on_cuda dst tgt)
    if(TARGET _gridtools_cuda)
        _gt_depends_on(result ${tgt} _gridtools_cuda)
        set(${dst} ${result} PARENT_SCOPE)
    else()
        set(${dst} OFF PARENT_SCOPE)
    endif()
endfunction()

function(_gt_depends_on_nvcc dst tgt)
    if(TARGET _gridtools_nvcc)
        _gt_depends_on(result ${tgt} _gridtools_nvcc)
        set(${dst} ${result} PARENT_SCOPE)
    else()
        set(${dst} OFF PARENT_SCOPE)
    endif()
endfunction()

function(_gt_depends_on_gridtools dst tgt)
    if(TARGET gridtools)
        _gt_depends_on(result ${tgt} gridtools)
    elseif(TARGET GridTools::gridtools)
        _gt_depends_on(result ${tgt} GridTools::gridtools)
    else()
        message(AUTHOR_WARNING "No GridTools targets are defined. This is a GridTools CMake implementation issue.")
    endif()
    if(result)
        set(${dst} TRUE PARENT_SCOPE)
    else()
        set(${dst} FALSE PARENT_SCOPE)
    endif()
endfunction()

set(_GT_INCLUDER_IN ${CMAKE_CURRENT_LIST_DIR}/includer.in)

function(_gt_convert_to_cuda_source dst srcfile)
    get_filename_component(extension ${srcfile} LAST_EXT)
    if(extension STREQUAL ".cu")
        set(${dst} ${srcfile} PARENT_SCOPE)
    else()
        set(INCLUDER_SRC ${CMAKE_CURRENT_SOURCE_DIR}/${srcfile})
        configure_file(${_GT_INCLUDER_IN} ${srcfile}.cu)
        set(${dst} ${CMAKE_CURRENT_BINARY_DIR}/${srcfile}.cu PARENT_SCOPE)
    endif()
endfunction()

function(_gt_convert_to_cxx_source srcfile)
    get_filename_component(extension ${srcfile} LAST_EXT)
    if(extension STREQUAL ".cu")
        set_source_files_properties(${srcfile} PROPERTIES LANGUAGE CXX)
    endif()
endfunction()

function(_gt_normalize_source_names lib dst)
    _gt_depends_on_nvcc(nvcc_cuda ${lib})
    if (nvcc_cuda)
        foreach(srcfile IN LISTS ARGN)
            _gt_convert_to_cuda_source(converted ${srcfile})
            list(APPEND acc ${converted})
        endforeach()
        set(${dst} ${acc} PARENT_SCOPE)
    else()
        foreach(srcfile IN LISTS ARGN)
            _gt_convert_to_cxx_source(${srcfile})
        endforeach()
        set(${dst} ${ARGN} PARENT_SCOPE)
    endif()
endfunction()

function(_gt_normalize_target_sources tgt)
    # SOURCES
    get_target_property(_sources ${tgt} SOURCES)
    if(_sources)
        _gt_normalize_source_names(${tgt} normalized_sources ${_sources})
        set_target_properties(${tgt} PROPERTIES SOURCES ${normalized_sources})
    endif()
    # INTERFACE_SOURCES
    get_target_property(_interface_sources ${tgt} INTERFACE_SOURCES)
    if(_interface_sources)
        _gt_normalize_source_names(${tgt} normalized_sources ${_interface_sources})
        set_target_properties(${tgt} PROPERTIES INTERFACE_SOURCES ${normalized_sources})
    endif()
endfunction()

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
macro(_gt_setup_targets _config_mode clang_cuda_mode)
    set(GT_AVAILABLE_TARGETS)

    include(detect_features)
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
    include(workaround_mpi)
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
            find_package(CUDAToolkit)
            if(CUDAToolkit_FOUND)
                target_link_libraries(_gridtools_cuda INTERFACE CUDA::cudart)
            else()
                message(FATAL_ERROR "NVCC was found, but the CUDAToolkit was not found."
                    "This should not happen. Please report this issue at https://github.com/GridTools/gridtools."
                )
            endif()
        elseif(type STREQUAL Clang-CUDA)
            set(_gt_setup_root_dir ${CUDAToolkit_BIN_DIR}/..)
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

    _gt_add_library(${_config_mode} storage_cpu_kfirst)
    target_link_libraries(${_gt_namespace}storage_cpu_kfirst INTERFACE ${_gt_namespace}gridtools)

    _gt_add_library(${_config_mode} storage_cpu_ifirst)
    target_link_libraries(${_gt_namespace}storage_cpu_ifirst INTERFACE ${_gt_namespace}gridtools)

    _gt_add_library(${_config_mode} stencil_naive)
    target_link_libraries(${_gt_namespace}stencil_naive INTERFACE ${_gt_namespace}gridtools)

    set(GT_STENCILS naive)
    set(GT_STORAGES cpu_kfirst cpu_ifirst)
    set(GT_GCL_ARCHS)

    if (TARGET _gridtools_cuda)
        _gt_add_library(${_config_mode} stencil_gpu)
        target_link_libraries(${_gt_namespace}stencil_gpu INTERFACE ${_gt_namespace}gridtools _gridtools_cuda)
        list(APPEND GT_STENCILS gpu)

        _gt_add_library(${_config_mode} stencil_gpu_horizontal)
        target_link_libraries(${_gt_namespace}stencil_gpu_horizontal INTERFACE ${_gt_namespace}gridtools _gridtools_cuda)
        list(APPEND GT_STENCILS gpu_horizontal)

        if(MPI_CXX_FOUND)
            option(GT_GCL_GPU "Disable if your MPI implementation is not CUDA-aware" ON)
            if(GT_GCL_GPU)
                _gt_add_library(${_config_mode} gcl_gpu)
                target_link_libraries(${_gt_namespace}gcl_gpu INTERFACE ${_gt_namespace}gridtools _gridtools_cuda MPI::MPI_CXX)
            endif()
        endif()

        _gt_add_library(${_config_mode} boundaries_gpu)
        target_link_libraries(${_gt_namespace}boundaries_gpu INTERFACE ${_gt_namespace}gridtools _gridtools_cuda)

        _gt_add_library(${_config_mode} layout_transformation_gpu)
        target_link_libraries(${_gt_namespace}layout_transformation_gpu INTERFACE ${_gt_namespace}gridtools _gridtools_cuda)

        list(APPEND GT_GCL_ARCHS gpu)
    endif()

    find_package(CUDAToolkit)
    if(CUDAToolkit_FOUND)
        _gt_add_library(${_config_mode} storage_gpu)
        target_link_libraries(${_gt_namespace}storage_gpu INTERFACE ${_gt_namespace}gridtools CUDA::cudart)
        list(APPEND GT_STORAGES gpu)
    endif()

    if (OpenMP_CXX_FOUND)
        _gt_add_library(${_config_mode} stencil_cpu_kfirst)
        target_link_libraries(${_gt_namespace}stencil_cpu_kfirst INTERFACE ${_gt_namespace}gridtools OpenMP::OpenMP_CXX)

        _gt_add_library(${_config_mode} stencil_cpu_ifirst)
        target_link_libraries(${_gt_namespace}stencil_cpu_ifirst INTERFACE ${_gt_namespace}gridtools OpenMP::OpenMP_CXX)

        if(MPI_CXX_FOUND)
            _gt_add_library(${_config_mode} gcl_cpu)
            target_link_libraries(${_gt_namespace}gcl_cpu INTERFACE ${_gt_namespace}gridtools OpenMP::OpenMP_CXX MPI::MPI_CXX)
        endif()
        _gt_add_library(${_config_mode} boundaries_cpu)
        target_link_libraries(${_gt_namespace}boundaries_cpu INTERFACE ${_gt_namespace}gridtools OpenMP::OpenMP_CXX)

        _gt_add_library(${_config_mode} layout_transformation_cpu)
        target_link_libraries(${_gt_namespace}layout_transformation_cpu INTERFACE ${_gt_namespace}gridtools OpenMP::OpenMP_CXX)

        list(APPEND GT_GCL_ARCHS cpu)

        list(APPEND GT_STENCILS cpu_kfirst cpu_ifirst)

    endif()

    # Setup HPX
    find_package(HPX QUIET NO_MODULE)
    if (HPX_FOUND)
        # Add a stencil target for HPX backend
        _gt_add_library(${_config_mode} stencil_cpu_ifirst_hpx)
        target_link_libraries(${_gt_namespace}stencil_cpu_ifirst_hpx INTERFACE ${_gt_namespace}gridtools HPX::hpx_no_wrap_main)
        list(APPEND GT_STENCILS cpu_ifirst_hpx)
    endif(HPX_FOUND)

endmacro()

function(_gt_print_configuration_summary)
    message(STATUS "GridTools configuration summary")
    message(STATUS "  Available targets: ${GT_AVAILABLE_TARGETS}")
    message(STATUS "  GPU mode: ${GT_CUDA_TYPE}") # Note that the CMake tests rely on the string "GPU mode: <mode>"
                                                  # to be present in the CMake configure output
endfunction()

# gridtools_set_gpu_arch_on_target()
# Sets the cuda architecture using the the compiler-dependant flag.
# Ignores the call if the target doesn't link to _gridtools_cuda.
# Note: Only set the architecture using this function, otherwise your setup might not be portable between Clang-CUDA
#       and nvcc.
function(gridtools_set_gpu_arch_on_target tgt arch)
    if(arch)
        _gt_depends_on_cuda(need_cuda ${tgt})
        if(need_cuda)
            _gt_depends_on_nvcc(need_nvcc ${tgt})
            if(need_nvcc)
                target_compile_options(${tgt} PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:${GT_CUDA_ARCH_FLAG}=${arch}>)
            else()
                target_compile_options(${tgt} PUBLIC ${GT_CUDA_ARCH_FLAG}=${arch})
            endif()
        endif()
    endif()
endfunction()

# gridtools_setup_target()
# Applying this function to a target will allow the same .cpp or .cu file to be compiled with CUDA and CXX in the same
# CMake project.
# Example: Compile a file copy_stencil.cpp for CUDA with stencil_cuda and for CPU with stencil_cpu_ifirst. In plain CMake this
# is not possible as you need to specify exactly one language to the file (either implicitly by file suffix or
# explicitly by setting the language). This function will wrap .cpp files in a .cu if the given target links to
# _gridtools_cuda.
function(gridtools_setup_target tgt)
    set(options)
    set(one_value_args CUDA_ARCH)
    set(multi_value_args)
    cmake_parse_arguments(ARGS "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

    _gt_depends_on_gridtools(links_to_a_gt_backend ${tgt})
    if(NOT links_to_a_gt_backend)
        message(FATAL_ERROR "gridtools_setup_target() needs to be called after a backend library is linked to the target")
    endif()
    _gt_normalize_target_sources(${tgt})
    gridtools_set_gpu_arch_on_target(${tgt} "${ARGS_CUDA_ARCH}")
endfunction()
