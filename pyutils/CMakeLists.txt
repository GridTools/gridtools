string(TOLOWER "${CMAKE_BUILD_TYPE}" PERFTEST_BUILD_TYPE)

if(GT_ENABLE_TARGET_CUDA)
    set(PERFTEST_COMPILER "${CMAKE_CUDA_COMPILER}")
else()
    set(PERFTEST_COMPILER "${CMAKE_CXX_COMPILER}")
endif()

if(GT_SINGLE_PRECISION)
    set(PERFTEST_PRECISION "float")
else()
    set(PERFTEST_PRECISION "double")
endif()

if(STRUCTURED_GRIDS)
    set(PERFTEST_GRID "structured")
else()
    set(PERFTEST_GRID "icosahedral")
endif()

set (ENABLED_BACKENDS )
if(GT_ENABLE_TARGET_X86)
    list(APPEND PERFTEST_BACKEND "x86")
endif()

if(GT_ENABLE_TARGET_CUDA)
    list(APPEND PERFTEST_BACKEND "cuda")
endif()

if(GT_ENABLE_TARGET_MC)
    list(APPEND PERFTEST_BACKEND "mc")
endif()
list(LENGTH PERFTEST_BACKEND PERFTEST_BACKEND_COUNT)
if(${PERFTEST_BACKEND_COUNT} GREATER 1)
    message(FATAL_ERROR "Performance tests can only be enabled for a single backend")
endif()

file(COPY . DESTINATION . FILES_MATCHING PATTERN *.py)
configure_file(perftest/buildinfo.py.in perftest/buildinfo.py)
