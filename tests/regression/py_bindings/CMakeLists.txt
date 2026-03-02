if (NOT ${GT_TESTS_ENABLE_PYTHON_TESTS})
    return()
endif()
if (NOT ${Python_Development_FOUND} OR NOT ${Python_NumPy_FOUND})
    return()
endif()

include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.10.0
)

FetchContent_MakeAvailable(pybind11)

pybind11_add_module(py_implementation implementation.cpp)

target_link_libraries(py_implementation PRIVATE gridtools)

add_test(NAME py_bindings COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/driver.py ${GT_CUDA_TYPE})
