string(TOLOWER "${CMAKE_BUILD_TYPE}" PERFTEST_BUILD_TYPE)

if(GT_ENABLE_BACKEND_CUDA)
    set(PERFTEST_COMPILER "${CMAKE_CUDA_COMPILER}")
else()
    set(PERFTEST_COMPILER "${CMAKE_CXX_COMPILER}")
endif()

if(GT_SINGLE_PRECISION)
    set(PERFTEST_PRECISION "float")
else()
    set(PERFTEST_PRECISION "double")
endif()

if(GT_TESTS_ICOSAHEDRAL_GRID)
    set(PERFTEST_GRID "icosahedral")
else()
    set(PERFTEST_GRID "structured")
endif()

set (ENABLED_BACKENDS )
if(GT_ENABLE_BACKEND_X86)
    list(APPEND PERFTEST_BACKEND "x86")
endif()

if(GT_ENABLE_BACKEND_CUDA)
    list(APPEND PERFTEST_BACKEND "cuda")
endif()

if(GT_ENABLE_BACKEND_MC)
    list(APPEND PERFTEST_BACKEND "mc")
endif()
list(LENGTH PERFTEST_BACKEND PERFTEST_BACKEND_COUNT)
if(${PERFTEST_BACKEND_COUNT} GREATER 1)
    message(FATAL_ERROR "Performance tests can only be enabled for a single backend")
endif()

file(COPY . DESTINATION . FILES_MATCHING PATTERN *.py)
configure_file(perftest/buildinfo.py.in perftest/buildinfo.py)
