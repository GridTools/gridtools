# Try to find PAPI wrap headers and libraries.
#
# Usage of this module as follows:
#
# find_package(PapiWrap)
#
# Variables used by this module, they can change the default behaviour and need
# to be set before calling find_package:
#
# PAPI_WRAP_PREFIX Set this variable to the root installation of
# papi_wrap if the module has problems finding the
# proper installation path.
#
# Variables defined by this module:
#
# PAPI_WRAP_FOUND System has PAPI libraries and headers
# PAPI_WRAP_LIBRARIES The PAPI library
# PAPI_WRAP_INCLUDE_DIRS The location of PAPI headers

find_path(PAPI_WRAP_PREFIX
    NAMES papi_wrap.h
)

find_library(PAPI_WRAP_LIBRARIES
    # Pick the static library first for easier run-time linking.
    NAMES libpapi_wrap.a papi_wrap
    HINTS ${PAPI_WRAP_PREFIX}/lib ${HILTIDEPS}/lib
)

find_path(PAPI_WRAP_INCLUDE_DIRS
    NAMES papi_wrap.h
    HINTS ${PAPI_WRAP_PREFIX} ${HILTIDEPS}
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PAPI_WRAP DEFAULT_MSG
    PAPI_WRAP_LIBRARIES
    PAPI_WRAP_INCLUDE_DIRS
)

mark_as_advanced(
    PAPI_WRAP_PREFIX_DIRS
    PAPI_WRAP_LIBRARIES
    PAPI_WRAP_INCLUDE_DIRS
)