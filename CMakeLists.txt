cmake_minimum_required(VERSION 3.14.5)
cmake_policy(VERSION 3.14.5)

file(STRINGS "version.txt" __GT_VERSION)
project(GridTools VERSION ${__GT_VERSION} LANGUAGES CXX)

message(STATUS "Configuring ${PROJECT_NAME} v${__GT_VERSION}" )

unset(__GT_VERSION)

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_EXTENSIONS OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

include(CMakePrintHelpers)
include(flags)
include(detect_features)

set(REQUIRED_BOOST_VERSION 1.58)
find_package(Boost ${REQUIRED_BOOST_VERSION} REQUIRED)

if(NOT (CMAKE_CXX_COMPILER_ID STREQUAL AppleClang AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 11.0))
    find_package(OpenMP COMPONENTS CXX)
endif()

# detect what cuda do we have
detect_cuda_type(GT_CUDA_TYPE "${GT_CLANG_CUDA_MODE}")
mark_as_advanced(GT_CUDA_TYPE)

# Enable cuda language
# Additionally add the gridtools_nvcc proxy
if (GT_CUDA_TYPE STREQUAL NVCC-CUDA)
    enable_language(CUDA)
    add_library(gridtools_nvcc INTERFACE)
    # allow to call constexpr __host__ from constexpr __device__, e.g. call std::max in constexpr context
    target_compile_options(gridtools_nvcc INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
endif()

find_package(MPI COMPONENTS CXX)

if (GT_CUDA_TYPE STREQUAL NVCC-CUDA)
    set(GT_CUDA_ARCH_FLAG -arch)
elseif (GT_CUDA_TYPE STREQUAL Clang-CUDA)
    set(GT_CUDA_ARCH_FLAG --cuda-gpu-arch)
elseif (GT_CUDA_TYPE STREQUAL HIPCC-AMDGPU)
    set(GT_CUDA_ARCH_FLAG --amdgpu-target)
endif()

function(gridtools_cuda_setup type)
    if (type STREQUAL NVCC-CUDA)
        target_link_libraries(gridtools_cuda INTERFACE gridtools_nvcc)
        find_library(cudart cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
        set(GT_CUDA_LIBRARIES ${cudart} PARENT_SCOPE)
        set(GT_CUDA_INCLUDE_DIRS ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} PARENT_SCOPE)
        message(STATUS "GridTools GPU mode: CUDA with NVCC")
    elseif(type STREQUAL Clang-CUDA)
        set(root_dir ${CUDAToolkit_BIN_DIR}/..)
        set(GT_CUDA_LIBRARIES ${CUDA_LIBRARIES} PARENT_SCOPE)
        set(GT_CUDA_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS} PARENT_SCOPE)
        target_compile_options(gridtools_cuda INTERFACE -xcuda --cuda-path=${root_dir})
        target_link_libraries(gridtools_cuda INTERFACE CUDA::cudart)
        message(STATUS "GridTools GPU mode: CUDA with clang")
    elseif(type STREQUAL HIPCC-AMDGPU)
        target_compile_options(gridtools_cuda INTERFACE -xhip)
        message(STATUS "GridTools GPU mode: HIPCC on clang")
    endif()
endfunction()

if (GT_CUDA_TYPE)
    add_library(gridtools_cuda INTERFACE)
    gridtools_cuda_setup(${GT_CUDA_TYPE})
else()
    message(STATUS "No CUDA- or HIP-capable compiler detected.")
endif()

add_library(gridtools INTERFACE)
target_compile_features(gridtools INTERFACE cxx_std_14)
target_link_libraries(gridtools INTERFACE Boost::boost)
target_compile_definitions(gridtools INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:BOOST_PP_VARIADICS=1>)
target_include_directories(gridtools
        INTERFACE
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/>
        $<INSTALL_INTERFACE:include>
        )

add_library(GridTools::gridtools ALIAS gridtools)
install(TARGETS gridtools EXPORT GridToolsTargets
        INCLUDES DESTINATION include
        )

add_library(storage_x86 INTERFACE)
target_link_libraries(storage_x86 INTERFACE gridtools)

add_library(storage_mc INTERFACE)
target_link_libraries(storage_mc INTERFACE gridtools)

add_library(backend_naive INTERFACE)
target_link_libraries(backend_naive INTERFACE gridtools)

set(GT_BACKENDS naive)
set(GT_ICO_BACKENDS naive)
set(GT_STORAGES x86 mc)
set(GT_GCL_ARCHS)

if (TARGET gridtools_cuda)
    add_library(backend_cuda INTERFACE)
    target_link_libraries(backend_cuda INTERFACE gridtools gridtools_cuda)
    list(APPEND GT_BACKENDS cuda)
    list(APPEND GT_ICO_BACKENDS cuda)

    add_library(storage_cuda INTERFACE)
    target_link_libraries(storage_cuda INTERFACE gridtools gridtools_cuda)
    list(APPEND GT_STORAGES cuda)

    if(MPI_CXX_FOUND)
        add_library(gcl_gpu INTERFACE)
        target_link_libraries(gcl_gpu INTERFACE gridtools gridtools_cuda MPI::MPI_CXX)
    endif()

    add_library(bc_gpu INTERFACE)
    target_link_libraries(bc_gpu INTERFACE gridtools gridtools_cuda)

    add_library(layout_transformation_gpu INTERFACE)
    target_link_libraries(layout_transformation_gpu INTERFACE gridtools gridtools_cuda)

    list(APPEND GT_GCL_ARCHS gpu)
endif()

if (OpenMP_CXX_FOUND)
    add_library(backend_x86 INTERFACE)
    target_link_libraries(backend_x86 INTERFACE gridtools OpenMP::OpenMP_CXX)

    add_library(backend_mc INTERFACE)
    target_link_libraries(backend_mc INTERFACE gridtools OpenMP::OpenMP_CXX)

    if(MPI_CXX_FOUND)
        add_library(gcl_cpu INTERFACE)
        target_link_libraries(gcl_cpu INTERFACE gridtools OpenMP::OpenMP_CXX MPI::MPI_CXX)
    endif()
    add_library(bc_cpu INTERFACE)
    target_link_libraries(bc_cpu INTERFACE gridtools OpenMP::OpenMP_CXX)

    add_library(layout_transformation_cpu INTERFACE)
    target_link_libraries(layout_transformation_cpu INTERFACE gridtools OpenMP::OpenMP_CXX)

    list(APPEND GT_GCL_ARCHS cpu)

    list(APPEND GT_BACKENDS x86 mc)
    list(APPEND GT_ICO_BACKENDS x86)
endif()

function(gridtools_depends_on dst lib dependency)
    if (NOT TARGET ${dependency})
        return()
    elseif (lib STREQUAL dependency)
        set(${dst} ON PARENT_SCOPE)
    elseif (TARGET ${lib})
        get_target_property(deps ${lib} INTERFACE_LINK_LIBRARIES)
        if (deps)
            foreach(dep IN LISTS deps)
                gridtools_depends_on(child ${dep} ${dependency})
                if (child)
                    set(${dst} ON PARENT_SCOPE)
                    return()
                endif()
            endforeach()
        endif()
    endif()
endfunction()

set(_GT_INCLUDER_IN ${CMAKE_CURRENT_LIST_DIR}/includer.in)

function(gridtools_convert_to_cuda_source dst srcfile)
    get_filename_component(extension ${srcfile} LAST_EXT)
    if(extension STREQUAL ".cu")
        set(${dst} ${srcfile} PARENT_SCOPE)
    else()
        set(INCLUDER_SRC ${CMAKE_CURRENT_SOURCE_DIR}/${srcfile})
        configure_file(${_GT_INCLUDER_IN} ${srcfile}.cu)
        set(${dst} ${CMAKE_CURRENT_BINARY_DIR}/${srcfile}.cu PARENT_SCOPE)
    endif()
endfunction()

function(gridtools_convert_to_cxx_source srcfile)
    get_filename_component(extension ${srcfile} LAST_EXT)
    if(extension STREQUAL ".cu")
        set_source_files_properties(${srcfile} PROPERTIES LANGUAGE CXX)
    endif()
endfunction()

function(gridtools_normalize_source_names lib dst)
    gridtools_depends_on(nvcc_cuda ${lib} gridtools_nvcc)
    if (nvcc_cuda)
        foreach(srcfile IN LISTS ARGN)
            gridtools_convert_to_cuda_source(converted ${srcfile})
            list(APPEND acc ${converted})
        endforeach()
        set(${dst} ${acc} PARENT_SCOPE)
    else()
        foreach(srcfile IN LISTS ARGN)
            gridtools_convert_to_cxx_source(${srcfile})
        endforeach()
        set(${dst} ${ARGN} PARENT_SCOPE)
    endif()
endfunction()

function(gridtools_setup_target tgt)
    set(options)
    set(one_value_args CUDA_ARCH)
    set(multi_value_args SOURCES LIBRARIES)
    cmake_parse_arguments(ARGS "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})
    target_link_libraries(${tgt} PUBLIC ${ARGS_LIBRARIES})
    gridtools_normalize_source_names(${tgt} normalized_sources ${ARGS_SOURCES})
    target_sources(${tgt} PRIVATE ${normalized_sources})
    if (ARGS_CUDA_ARCH)
        gridtools_depends_on(need_cuda ${tgt} gridtools_cuda)
        if (need_cuda)
            target_compile_options(${tgt} PUBLIC ${GT_CUDA_ARCH_FLAG}=${ARGS_CUDA_ARCH})
        endif()
    endif()
endfunction()

function(gridtools_add_executable tgt)
    add_executable(${tgt})
    gridtools_setup_target(${tgt} ${ARGN})
endfunction()

function(gridtools_add_library tgt)
    add_library(${tgt})
    gridtools_setup_target(${tgt} ${ARGN})
endfunction()

if (CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    # add a target to generate API documentation with Doxygen
    find_package(Doxygen)
    if(DOXYGEN_FOUND)
        configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
        add_custom_target(doc ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile WORKING_DIRECTORY
                ${CMAKE_CURRENT_BINARY_DIR} COMMENT "Generating API documentation with Doxygen" VERBATIM)
    endif()

    include(CTest)
    if (BUILD_TESTING)
        if (TARGET MPI::MPI_CXX)
            include(workaround_mpiexec)
            _fix_mpi_exec()
        endif()

        if (TARGET gridtools_cuda)
            if (DEFINED ENV{CUDA_ARCH})
                set(GT_CUDA_ARCH_INIT $ENV{CUDA_ARCH})
            else()
                set(GT_CUDA_ARCH_INIT "sm_35")
            endif()
            set(GT_CUDA_ARCH "${GT_CUDA_ARCH_INIT}" CACHE STRING "Compute capability for CUDA used for tests")
        endif()
        include(detect_test_features)
        detect_c_compiler()
        detect_fortran_compiler()

        add_subdirectory(tests)
        add_subdirectory(docs_src/manuals/getting_started)
    endif()

    # ===============
    # examples
    # ===============
    if(GT_INSTALL_EXAMPLES)
        add_subdirectory(examples)
    endif()

    # ===============
    # Downloads the gridtools_experimental repository
    # ===============
    if(GT_ENABLE_EXPERIMENTAL_REPOSITORY)
        include(gridtools_experimental/download_gridtools_experimental)
    endif()

    # ===============
    # Python scripts
    # ===============
    if(GT_ENABLE_PYUTILS)
        add_subdirectory(pyutils)
    endif()
endif()

# absolute path to avoid collision with "export" from other packages
include(${CMAKE_CURRENT_LIST_DIR}/cmake/export.cmake)
