include(FetchContent)
FetchContent_Declare(
        cpp_bindgen
        GIT_REPOSITORY https://github.com/GridTools/cpp_bindgen.git
        GIT_TAG        v1.0.1
)

set(build_testing_ ${BUILD_TESTING})
set(BUILD_TESTING OFF)
FetchContent_MakeAvailable(cpp_bindgen)
set(BUILD_TESTING ${build_testing_})

gridtools_add_unit_test(test_fortran_array_adapter
        SOURCES test_fortran_array_adapter.cpp
        LIBRARIES cpp_bindgen_interface
        NO_NVCC)


if (${GT_TESTS_ENABLE_PYTHON_TESTS})
        if (${Python_Development_FOUND})
                FetchContent_Declare(
                        nanobind
                        GIT_REPOSITORY https://github.com/wjakob/nanobind.git
                        GIT_TAG        v2.0.0
                )
                FetchContent_MakeAvailable(nanobind)
                nanobind_build_library(nanobind-static)

                gridtools_add_unit_test(test_nanobind_adapter
                        SOURCES test_nanobind_adapter.cpp
                        LIBRARIES nanobind-static Python::Python
                        NO_NVCC)
                nanobind_compile_options(test_nanobind_adapter)
                nanobind_link_options(test_nanobind_adapter)
        endif()
endif()
