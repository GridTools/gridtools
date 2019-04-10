string(TOLOWER "${CMAKE_BUILD_TYPE}" PYUTILS_BUILD_TYPE)

if(GT_ENABLE_BACKEND_CUDA)
    set(PYUTILS_COMPILER "${CMAKE_CUDA_COMPILER}")
else()
    set(PYUTILS_COMPILER "${CMAKE_CXX_COMPILER}")
endif()

if(GT_SINGLE_PRECISION)
    set(PYUTILS_PRECISION "float")
else()
    set(PYUTILS_PRECISION "double")
endif()

if(GT_TESTS_ICOSAHEDRAL_GRID)
    set(PYUTILS_GRID "icosahedral")
else()
    set(PYUTILS_GRID "structured")
endif()

set (ENABLED_BACKENDS )
if(GT_ENABLE_BACKEND_X86)
    list(APPEND PYUTILS_BACKEND "x86")
endif()

if(GT_ENABLE_BACKEND_CUDA)
    list(APPEND PYUTILS_BACKEND "cuda")
endif()

if(GT_ENABLE_BACKEND_MC)
    list(APPEND PYUTILS_BACKEND "mc")
endif()

file(COPY . DESTINATION . FILES_MATCHING PATTERN *.py)
file(COPY . DESTINATION . FILES_MATCHING PATTERN *.sh)
configure_file(pyutils/buildinfo.py.in pyutils/buildinfo.py)
