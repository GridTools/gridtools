string(TOLOWER "${CMAKE_BUILD_TYPE}" PERFTEST_BUILD_TYPE)

if(ENABLE_CUDA)
    set(PERFTEST_COMPILER "${CUDA_NVCC_EXECUTABLE}")
else()
    set(PERFTEST_COMPILER "${CMAKE_CXX_COMPILER}")
endif()

if(SINGLE_PRECISION)
    set(PERFTEST_PRECISION "float")
else()
    set(PERFTEST_PRECISION "double")
endif()

if(STRUCTURED_GRIDS)
    set(PERFTEST_GRID "structured")
else()
    set(PERFTEST_GRID "icosahedral")
endif()

if(ENABLE_HOST AND ENABLE_CUDA)
    message(FATAL_ERROR "Performance tests can only be enabled for a single backend")
endif()

if(ENABLE_HOST)
    set(PERFTEST_BACKEND "host")
endif()

if(ENABLE_CUDA)
    set(PERFTEST_BACKEND "cuda")
endif()

if(ENABLE_MIC)
    set(PERFTEST_BACKEND "mic")
endif()

file(COPY . DESTINATION . FILES_MATCHING PATTERN *.py)
configure_file(perftest/buildinfo.py.in perftest/buildinfo.py)
