
add_library(GridToolsTest INTERFACE)
target_link_libraries(GridToolsTest INTERFACE gridtools)
#target_compile_options(GridToolsTest INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:-arch=${GT_CUDA_ARCH}>)
target_include_directories(GridToolsTest INTERFACE include)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(GridToolsTest INTERFACE
            $<$<COMPILE_LANGUAGE:CXX>:-Wall
            -Wno-unknown-pragmas
            -Wno-sign-compare
            -Wno-unused-local-typedefs
            -Wno-attributes
            -Wno-unused-but-set-variable
            -Wno-unneeded-internal-declaration
            -Wno-unused-function>)
    target_compile_options(GridToolsTest INTERFACE
            "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler
            -Wall,-Wno-unknown-pragmas,-Wno-sign-compare,-Wno-attributes,-Wno-unused-but-set-variable,-Wno-unneeded-internal-declaration,-Wno-unused-function>")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "8.4.0")
        target_compile_options(GridToolsTest INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-value>)
        target_compile_options(GridToolsTest INTERFACE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler -Wno-unused-value>")
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

option(GT_TREAT_WARNINGS_AS_ERROR "Treat warnings as errors" OFF)
mark_as_advanced(GT_TREAT_WARNINGS_AS_ERROR)

if(GT_TREAT_WARNINGS_AS_ERROR)
    target_compile_options(GridToolsTest INTERFACE $<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-Werror>)
endif()

## cuda support ##
if(TARGET gridtools_nvcc)
    if(GT_TREAT_WARNINGS_AS_ERROR)
        # unfortunately we cannot treat all as warnings, we have to specify each warning; the only supported warning in CUDA8 is cross-execution-space-call
        # CUDA 9 adds deprecated-declarations (activated) and reorder (not activated)
        target_compile_options(GridToolsTest INTERFACE
                $<$<COMPILE_LANGUAGE:CUDA>:-Werror=cross-execution-space-call>
                $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=--warning-as-error>
                $<$<COMPILE_LANGUAGE:CUDA>:-Xnvlink=--warning-as-error>
                $<$<COMPILE_LANGUAGE:CUDA>:-Werror=deprecated-declarations>)
    endif()

    if (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS_EQUAL 9.2)
        # suppress because of warnings in GTest
        target_compile_options(GridToolsTest INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe=--diag_suppress=177>)
    endif()
endif()

function(gridtools_setup_test_target tgt)
    set(options)
    set(one_value_args CUDA_ARCH)
    set(multi_value_args SOURCES LIBRARIES)
    cmake_parse_arguments(ARGS "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})
    gridtools_setup_target(${tgt}
            SOURCES ${ARGS_SOURCES}
            LIBRARIES GridToolsTest ${ARGS_LIBRARIES}
            CUDA_ARCH ${GT_CUDA_ARCH})
endfunction()

function(gridtools_add_test_executable tgt)
    add_executable(${tgt})
    gridtools_setup_test_target(${tgt} ${ARGN})
endfunction()


function(gridtools_add_test_library tgt)
    add_library(${tgt})
    gridtools_setup_test_target(${tgt} ${ARGN})
endfunction()


####################################################################################
########################### GET GTEST LIBRARY ############################
####################################################################################

# include Threads manually before googletest such that we can properly apply the workaround
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package( Threads REQUIRED )
target_link_libraries( GridToolsTest INTERFACE Threads::Threads)
include(workaround_threads)
_fix_threads_flags()

# The gtest library needs to be built as static library to avoid RPATH issues,
# so make sure that the variable BUILD_SHARED_LIBS=OFF and restored back afterwards.
if( BUILD_SHARED_LIBS )
    set( _restore_BUILD_SHARED_LIBS ON )
    set( BUILD_SHARED_LIBS OFF )
endif()

include(FetchContent)
option(INSTALL_GTEST OFF) #TODO replace with set(INSTALL_GTEST OFF) with CMake >= 3.13
mark_as_advanced(INSTALL_GTEST)
FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        release-1.8.1
)
FetchContent_MakeAvailable(googletest)

if( _restore_BUILD_SHARED_LIBS )
    set( BUILD_SHARED_LIBS ON )
endif()

add_subdirectory(src)
add_subdirectory(regression)
add_subdirectory(unit_tests)
