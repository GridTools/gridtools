# _gt_depends_on()
# Note: This function relies on the unique name of a dependency. If libraries get a namespace prefix in exported
# context, this function may fail. Therefore, use of this function with care!
function(_gt_depends_on dst lib dependency)
    if (NOT TARGET ${dependency})
        set(${dst} OFF PARENT_SCOPE)
        return()
    elseif (lib STREQUAL dependency)
        set(${dst} ON PARENT_SCOPE)
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
endfunction()


function(_gt_depends_on_cuda dst tgt)
    _gt_depends_on(result ${tgt} _gridtools_cuda)
    set(${dst} ${result} PARENT_SCOPE)
endfunction()

function(_gt_depends_on_nvcc dst tgt)
    _gt_depends_on(result ${tgt} _gridtools_nvcc)
    set(${dst} ${result} PARENT_SCOPE)
endfunction()

function(_gt_depends_on_gridtools dst tgt)
    _gt_depends_on(result_no_ns ${tgt} gridtools)
    _gt_depends_on(result_ns ${tgt} GridTools::gridtools)
    if(result_no_ns OR result_ns)
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

# gridtools_set_gpu_arch_on_target()
# Sets the cuda architecture using the the compiler-dependant flag.
# Ignores the call if the target doesn't link to _gridtools_cuda.
# Note: Only set the architecture using this function, otherwise your setup might not be portable between Clang-CUDA
#       and nvcc.
function(gridtools_set_gpu_arch_on_target tgt arch)
    if(arch)
        _gt_depends_on_cuda(need_cuda ${tgt})
        if(need_cuda)
            target_compile_options(${tgt} PUBLIC ${GT_CUDA_ARCH_FLAG}=${arch})
        endif()
    endif()
endfunction()

# gridtools_setup_target()
# Applying this function to a target will allow the same .cpp or .cu file to be compiled with CUDA and CXX in the same
# CMake project.
# Example: Compile a file copy_stencil.cpp for CUDA with backend_cuda and for CPU with backend_mc. In plain CMake this
# is not possible as you need to specify exactly one language to the file (either implicitly by file suffix or
# explicitly by setting the language). This function will wrap .cpp files in a .cu if the given target links to
# _gridtools_cuda.
function(gridtools_setup_target tgt) # TODO this function needs a better name
    set(options)
    set(one_value_args CUDA_ARCH)
    set(multi_value_args)
    cmake_parse_arguments(ARGS "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

    # TODO check that we depend on a gridtools target
    _gt_depends_on_gridtools(links_to_a_gt_backend ${tgt})
    if(NOT links_to_a_gt_backend)
        message(FATAL_ERROR "gridtools_setup_target() needs to be called after a backend library is linked to the target")
    endif()
    _gt_normalize_target_sources(${tgt})
    gridtools_set_gpu_arch_on_target(${tgt} "${ARGS_CUDA_ARCH}")
endfunction()
