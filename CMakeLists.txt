cmake_minimum_required(VERSION 3.14.5)

cmake_policy(SET CMP0048 NEW)
file(STRINGS "version.txt" __GT_VERSION)
project(GridTools VERSION ${__GT_VERSION} LANGUAGES CXX)

message( STATUS "[GridTools] (${__GT_VERSION})" )

unset(__GT_VERSION)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

include(CTest)

# ===============
# Load some cmake modules.
# 1) define what flags should be available,

include(detect_features)
detect_cuda()
detect_openmp()
detect_mpi()

include(flags)

if(POLICY CMP0060)
    # Policy to avoid cmake to substitute libraries with paths and extensions with -l<libname>
    cmake_policy(SET CMP0060 NEW)
endif()

# ===============
# Load some cmake modules.
# 2) package by name finder
# 3) testing methods
# 4) build configurator (include path, compiler flags, etc.
# ===============
include(definitions)
if (BUILD_TESTING)
    include(testing)
endif()

include(FetchContent)
FetchContent_Declare(
    cpp_bindgen
    GIT_REPOSITORY https://github.com/GridTools/cpp_bindgen.git
    GIT_TAG        master
    )
# Enable legacy mode for CPP_BINDGEN for users of GT 1.0
# TODO remove in GT 2.0
set(CPP_BINDGEN_GT_LEGACY ON)
FetchContent_MakeAvailable(cpp_bindgen)

add_subdirectory(src)

# ===============
# tests
# ===============
if(BUILD_TESTING)
    add_subdirectory(regression)
    add_subdirectory(unit_tests)

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

# absolute path to avoid collision with "export" from other packages
include(${CMAKE_CURRENT_LIST_DIR}/cmake/export.cmake)
