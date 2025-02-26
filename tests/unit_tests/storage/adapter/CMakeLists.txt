if (${GT_TESTS_ENABLE_PYTHON_TESTS})
        if (${Python_Development_FOUND})
                FetchContent_Declare(
                        nanobind
                        GIT_REPOSITORY https://github.com/wjakob/nanobind.git
                        GIT_TAG        v2.1.0
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
