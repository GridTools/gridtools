if("${GT_CXX_STANDARD}" STREQUAL "c++14")
    set(GT_CXX_STANDARD_VALUE 14)
elseif("${GT_CXX_STANDARD}" STREQUAL "c++17")
    set(GT_CXX_STANDARD_VALUE 17)
else()
    message(FATAL_ERROR "Invalid argument for GT_CXX_STANDARD (`${GT_CXX_STANDARD}`)")
endif()

set(CMAKE_CXX_STANDARD ${GT_CXX_STANDARD_VALUE})
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD ${GT_CXX_STANDARD_VALUE})
set(CMAKE_CUDA_EXTENSIONS OFF)

add_library(gridtools INTERFACE)
add_library(GridTools::gridtools ALIAS gridtools)
target_compile_features(gridtools INTERFACE cxx_std_14)
target_include_directories(gridtools
    INTERFACE
      $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/>
      $<INSTALL_INTERFACE:include>
    )
install(TARGETS gridtools EXPORT GridToolsTargets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)
target_link_libraries(gridtools INTERFACE cpp_bindgen_interface)

set(REQUIRED_BOOST_VERSION 1.58)
find_package(Boost ${REQUIRED_BOOST_VERSION} REQUIRED)
target_link_libraries(gridtools INTERFACE Boost::boost)

if(OPENMP_AVAILABLE)
    target_link_libraries(gridtools INTERFACE OpenMP::OpenMP_CXX)
endif()

target_compile_definitions(gridtools INTERFACE BOOST_PP_VARIADICS=1)
if(CUDA_AVAILABLE)
    target_compile_definitions(gridtools INTERFACE GT_USE_GPU)
    if(GT_CUDA_COMPILATION_TYPE STREQUAL "NVCC-CUDA")
        if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_LESS 9.0)
            message(FATAL_ERROR "CUDA 8.X or lower is not supported")
        endif()

        # allow to call constexpr __host__ from constexpr __device__, e.g. call std::max in constexpr context
        target_compile_options(gridtools INTERFACE
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

        if(${GT_CXX_STANDARD} STREQUAL "c++17")
            message(FATAL_ERROR "c++17 is not supported for CUDA compilation")
        endif()
    elseif(GT_CUDA_COMPILATION_TYPE STREQUAL "Clang-CUDA")
        get_filename_component(cuda_bin_dir_ ${CMAKE_CUDA_COMPILER} DIRECTORY)
        get_filename_component(cuda_root_dir_ ${cuda_bin_dir_} DIRECTORY)
        set(clang_cuda_options_ -xcuda --cuda-gpu-arch=${GT_CUDA_ARCH} --cuda-path=${cuda_root_dir_})
        target_compile_options(gridtools INTERFACE $<$<COMPILE_LANGUAGE:CXX>:${clang_cuda_options_}>)
    elseif(GT_CUDA_COMPILATION_TYPE STREQUAL "HIPCC-AMDGPU")
        set(hipcc_options_ -xhip --amdgpu-target=${GT_CUDA_ARCH})
        target_compile_options(gridtools INTERFACE $<$<COMPILE_LANGUAGE:CXX>:${hipcc_options_}>)
        target_compile_definitions(gridtools INTERFACE GT_USE_HIP)
    endif()

    if(GT_CUDA_COMPILATION_TYPE MATCHES ".*CUDA")
        target_include_directories( gridtools INTERFACE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} )
        target_link_libraries( gridtools INTERFACE ${CUDA_CUDART_LIBRARY} )
    endif()
endif()

if(MPI_AVAILABLE)
    target_compile_definitions(gridtools INTERFACE GCL_MPI)
    if(GT_ENABLE_BACKEND_CUDA)
      target_compile_definitions(gridtools INTERFACE GCL_GPU)
    endif()
endif()

add_library(GridToolsTest INTERFACE)
target_link_libraries(GridToolsTest INTERFACE gridtools)
target_compile_definitions(GridToolsTest INTERFACE FUSION_MAX_VECTOR_SIZE=20)
target_compile_definitions(GridToolsTest INTERFACE FUSION_MAX_MAP_SIZE=20)
target_compile_options(GridToolsTest INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:-arch=${GT_CUDA_ARCH}>)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(GridToolsTest INTERFACE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wno-unknown-pragmas -Wno-sign-compare -Wno-unused-local-typedefs -Wno-attributes -Wno-unused-but-set-variable -Wno-unneeded-internal-declaration -Wno-unused-function>)
    target_compile_options(GridToolsTest INTERFACE
        "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler -Wall,-Wno-unknown-pragmas,-Wno-sign-compare,-Wno-attributes,-Wno-unused-but-set-variable,-Wno-unneeded-internal-declaration,-Wno-unused-function>")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "8.4.0")
        target_compile_options(GridToolsTest INTERFACE
            $<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-value>)
        target_compile_options(GridToolsTest INTERFACE
            "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler -Wno-unused-value>")
    endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "3.9.0")
        # attribute noalias has been added in clang 3.9.0
        target_compile_options(GridToolsTest INTERFACE
            $<$<COMPILE_LANGUAGE:CXX>:-Wno-unknown-attributes>)
        target_compile_options(GridToolsTest INTERFACE
            "SHELL:$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -Wno-unknown-attributes>")
    endif()
endif()

if(GT_TREAT_WARNINGS_AS_ERROR)
    target_compile_options(GridToolsTest INTERFACE $<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-Werror>)
endif()

# TESTS ONLY
if(GT_ENABLE_BACKEND_X86)
    add_library(GridToolsTestX86 INTERFACE)
    target_compile_definitions(GridToolsTestX86 INTERFACE GT_BACKEND_X86)
    target_link_libraries(GridToolsTestX86 INTERFACE GridToolsTest)
    target_compile_options(GridToolsTestX86 INTERFACE -march=native)
endif()

# TESTS ONLY
if(GT_ENABLE_BACKEND_NAIVE)
    add_library(GridToolsTestNAIVE INTERFACE)
    target_compile_definitions(GridToolsTestNAIVE INTERFACE GT_BACKEND_NAIVE)
    target_link_libraries(GridToolsTestNAIVE INTERFACE GridToolsTest)
    target_compile_options(GridToolsTestNAIVE INTERFACE -march=native)
endif()

## cuda support ##
if(GT_ENABLE_BACKEND_CUDA)
    if(GT_TREAT_WARNINGS_AS_ERROR)
        # unfortunately we cannot treat all as warnings, we have to specify each warning; the only supported warning in CUDA8 is cross-execution-space-call
        # CUDA 9 adds deprecated-declarations (activated) and reorder (not activated)
        target_compile_options(GridToolsTest INTERFACE
            $<$<COMPILE_LANGUAGE:CUDA>:-Werror=cross-execution-space-call>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=--warning-as-error>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xnvlink=--warning-as-error>
            $<$<COMPILE_LANGUAGE:CUDA>:-Werror=deprecated-declarations>)
    endif()

    if (GT_CUDA_COMPILATION_TYPE STREQUAL "NVCC-CUDA" AND CMAKE_CUDA_COMPILER_VERSION VERSION_LESS_EQUAL 9.2)
        # suppress because of warnings in GTest
        target_compile_options(GridToolsTest INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe=--diag_suppress=177>)
    endif()

    add_library(GridToolsTestCUDA INTERFACE)
    target_compile_definitions(GridToolsTestCUDA INTERFACE GT_BACKEND_CUDA)
    target_link_libraries(GridToolsTestCUDA INTERFACE GridToolsTest)
endif()

if(GT_ENABLE_BACKEND_MC)
    add_library(GridToolsTestMC INTERFACE)
    target_compile_definitions(GridToolsTestMC INTERFACE GT_BACKEND_MC)
    target_link_libraries(GridToolsTestMC INTERFACE GridToolsTest)
endif()

# TODO: Move to separate file?
if(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    # TODO add those flags to documentation (slightly improve performance)
    target_compile_options(GridToolsTest INTERFACE -qopt-subscript-in-range -qoverride-limits)
    # disable failed vectorization warnings for OpenMP SIMD loops
    target_compile_options(GridToolsTest INTERFACE -diag-disable=15518,15552)
endif()

## precision ##
if(GT_SINGLE_PRECISION)
    target_compile_definitions(GridToolsTest INTERFACE GT_FLOAT_PRECISION=4)
    message(STATUS "Compile tests in single precision")
else()
    target_compile_definitions(GridToolsTest INTERFACE GT_FLOAT_PRECISION=8)
   message(STATUS "Compile tests in double precision")
endif()

# add a target to generate API documentation with Doxygen
find_package(Doxygen)
if(DOXYGEN_FOUND)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
    add_custom_target(doc ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile WORKING_DIRECTORY
        ${CMAKE_CURRENT_BINARY_DIR} COMMENT "Generating API documentation with Doxygen" VERBATIM)
endif()

## test script generator ##
function(gridtools_add_test)
    set(options)
    set(one_value_args NAME)
    set(multi_value_args COMMAND LABELS)
    cmake_parse_arguments(__ "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})
    if(NOT ___LABELS)
        message(FATAL_ERROR "gridtools_add_test was called without LABELS")
    endif()
    string(REPLACE ";" " " command "${___COMMAND}")

    if(NOT ${TEST_USE_WRAPPERS_FOR_ALL_TESTS})
        add_test(NAME ${___NAME} COMMAND ${___COMMAND})
    else()
        add_test(
            NAME ${___NAME}
            COMMAND  ${MPITEST_EXECUTABLE} ${MPITEST_NUMPROC_FLAG} 1 ${MPITEST_PREFLAGS} ${___COMMAND} ${MPITEST_POSTFLAGS}
            )
    endif()
    set_tests_properties(${___NAME} PROPERTIES LABELS "${___LABELS}")
endfunction()

## test script generator for MPI tests ##
function(gridtools_add_mpi_test)
    set(options)
    set(one_value_args NAME NPROC)
    set(multi_value_args COMMAND LABELS)
    cmake_parse_arguments(__ "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})
    if(NOT ___NPROC)
        message(FATAL_ERROR "gridtools_add_mpi_test was called without NPROC")
    endif()
    if(NOT ___LABELS)
        message(FATAL_ERROR "gridtools_add_mpi_test was called without LABELS")
    endif()
    string(REPLACE ";" " " command "${___COMMAND}")

    # Note: We use MPITEST_ instead of MPIEXEC_ because our own MPI_TEST_-variables are slurm-aware
    add_test(
        NAME ${___NAME}
        COMMAND  ${MPITEST_EXECUTABLE} ${MPITEST_NUMPROC_FLAG} ${___NPROC} ${MPITEST_PREFLAGS} ${___COMMAND} ${MPITEST_POSTFLAGS}
        )
    set_tests_properties(${___NAME} PROPERTIES LABELS "${___LABELS}")
    set_tests_properties(${___NAME} PROPERTIES PROCESSORS ${___NPROC})
endfunction()
