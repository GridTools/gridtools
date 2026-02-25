if (${GT_TESTS_ENABLE_PYTHON_TESTS})
        if (${Python_Development_FOUND})
                FetchContent_Declare(
                        nanobind
                        GIT_REPOSITORY https://github.com/wjakob/nanobind.git
                        GIT_TAG        v2.1.0
                )
                FetchContent_MakeAvailable(nanobind)
                nanobind_build_library(nanobind-static)
                # Nvhpc does not support -ffunction-sections/-fdata-sections and
                # -Wl,--gc-sections, remove those flags from the target.
                if (CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
                    foreach(opts COMPILE_OPTIONS INTERFACE_COMPILE_OPTIONS)
                        get_target_property(nanonbind_opts nanobind-static ${opts})
                        if (nanonbind_opts)
                            list(FILTER nanonbind_opts EXCLUDE REGEX "ffunction-sections|fdata-sections")
                            set_target_properties(nanobind-static PROPERTIES ${opts} "${nanonbind_opts}")
                        endif()
                    endforeach()
                    foreach(opts LINK_OPTIONS INTERFACE_LINK_OPTIONS)
                        get_target_property(nanobind_opts nanobind-static ${opts})
                        if (nanobind_opts)
                            list(FILTER nanobind_opts EXCLUDE REGEX "gc-sections")
                            set_target_properties(nanobind-static PROPERTIES ${opts} "${nanobind_opts}")
                        endif()
                    endforeach()
                endif()

                gridtools_add_unit_test(test_nanobind_adapter
                        SOURCES test_nanobind_adapter.cpp
                        LIBRARIES nanobind-static Python::Python
                        NO_NVCC)
                nanobind_compile_options(test_nanobind_adapter)
                nanobind_link_options(test_nanobind_adapter)
        endif()
endif()
