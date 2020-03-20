cmake_minimum_required(VERSION 3.14.5)
cmake_policy(VERSION 3.14.5)

file(STRINGS "version.txt" __GT_VERSION)
project(GridTools VERSION ${__GT_VERSION} LANGUAGES CXX)

message(STATUS "Configuring ${PROJECT_NAME} v${__GT_VERSION}" )

unset(__GT_VERSION)

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_EXTENSIONS OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/public" "${CMAKE_CURRENT_LIST_DIR}/cmake/internal")

include(CMakePrintHelpers)
include(flags)

set(REQUIRED_BOOST_VERSION 1.58)
find_package(Boost ${REQUIRED_BOOST_VERSION} REQUIRED)

# setup main gridtools target
add_library(gridtools INTERFACE)
add_library(GridTools::gridtools ALIAS gridtools)
target_compile_features(gridtools INTERFACE cxx_std_14)
target_link_libraries(gridtools INTERFACE Boost::boost)
target_compile_definitions(gridtools INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:BOOST_PP_VARIADICS=1>)
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

include(gridtools_setup_targets)
gridtools_setup_targets(FALSE "${GT_CLANG_CUDA_MODE}")

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    # add a target to generate API documentation with Doxygen
    find_package(Doxygen)
    if(DOXYGEN_FOUND)
        configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
        add_custom_target(doc ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile WORKING_DIRECTORY
                ${CMAKE_CURRENT_BINARY_DIR} COMMENT "Generating API documentation with Doxygen" VERBATIM)
    endif()

    # Python scripts
    add_subdirectory(pyutils)
endif()

# Enable tests
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    include(CTest)
else()
    option(GT_BUILD_TESTING "Build GridTools tests" OFF)
    if(GT_BUILD_TESTING)
        include(CTest)
    endif()
     # Note that the following is setting a local variable BUILD_TESTING, not affecting the setting of a super-project
    set(BUILD_TESTING ${GT_BUILD_TESTING})
endif()
if(BUILD_TESTING)
    if(MPI_CXX_FOUND)
        include(workaround_mpiexec)
        _fix_mpi_exec()
    endif()

    if(TARGET gridtools_cuda)
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

# examples
if(GT_INSTALL_EXAMPLES)
    add_subdirectory(examples)
endif()

# absolute path to avoid collision with "export" from other packages
include(${CMAKE_CURRENT_LIST_DIR}/cmake/internal/export.cmake)
