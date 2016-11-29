##===------------------------------------------------------------------------------*- CMake -*-===##
##
##                                   S E R I A L B O X
##
## This file is distributed under terms of BSD license. 
## See LICENSE.txt for more information.
##
##===------------------------------------------------------------------------------------------===##

#.rst:
# FindSerialbox
# -------------
#
# Try to find Serialbox headers and libraries.
#
# Use this module by invoking find_package with the form::
#
#   find_package(Serialbox 
#    [version]                  # Minimum version e.g. 2.0
#    [REQUIRED]                 # Fail with error if Serialbox is not found
#    [COMPONENTS <languages>]   # Components of serialbox i.e the languages: C++, C, Fortran.
#                               # C++ is always ON.
#   )
#
# Example to find Serialbox headers and `shared` libraries for the C, C++ and Fortran interfaces:: 
#     
#   set(SERIALBOX_USE_SHARED_LIBS ON)
#   find_package(Serialbox REQUIRED COMPONENTS C++ C Fortran)
#
# The Serialbox module will look for the `exact` boost version used during compilation and append 
# the necessary libraries to the ``SERIALBOX_[LANGUAGE]_LIBRARIES`` variable. If Serialbox was 
# compiled with OpenSSL and/or NetCDF support, the necessary libraries will be appended as well. 
#
# Variables used by this module, they can change the default behaviour and need to be set before 
# calling find_package::
#
#   SERIALBOX_ROOT                  - Set this variable to the root installation of Serialbox if the 
#                                     module has problems finding the proper installation path.
#   SERIALBOX_USE_SHARED_LIBS       - Use the shared libraries (.so or .dylib) of Serialbox.
#   SERIALBOX_NO_EXTERNAL_LIBS      - Don't look for external libraries (Boost, NetCDF and OpenSSL)
#
# Variables defined by this module::
#
#   SERIALBOX_FOUND                 - True if headers and requested libraries were found
#   SERIALBOX_VERSION               - Version string of Serialbox (e.g "2.0.1")
#   SERIALBOX_INCLUDE_DIRS          - The location of the Serialbox headers (i.e to include the 
#                                     C Interface ${SERIALBOX_INCLUDE_DIRS}/serialbox-c/Serialbox.h)
#                                     possibly the boost headers and the Fortran mod files.
#   SERIALBOX_LIBRARY_DIR           - The location of the Serialbox libraries.
#   SERIALBOX_HAS_C                 - Serialbox was compiled with C support 
#   SERIALBOX_HAS_FORTRAN           - Serialbox was compiled with Fortran support
#   SERIALBOX_CXX_LIBRARIES         - The C++ libraries of Serialbox (libSerialboxCore) and 
#                                     possibly the external libraries.
#   SERIALBOX_C_LIBRARIES           - The C libraries of Serialbox (libSerialboxC) and 
#                                     possibly the external libraries.
#   SERIALBOX_FORTRAN_LIBRARIES     - The Fortran libraries of Serialbox (libSerialboxFortran) and 
#                                     possibly the external libraries.
#   SERIALBOX_PPSER                 - Path to the pp_ser.py script.
#   SERIALBOX_BOOST_VERSION         - Boost version used during compilation.
#   SERIALBOX_HAS_OPENSSL           - Serialbox was compiled with OpenSSL support.
#   SERIALBOX_HAS_NETCDF            - Serialbox was compiled with NetCDF support.
#

include(FindPackageHandleStandardArgs)

#
# Parse SERIALBOX_ROOT
#
set(SERIALBOX_ROOT_ENV $ENV{SERIALBOX_ROOT})
if(SERIALBOX_ROOT_ENV)
  set(SERIALBOX_ROOT ${SERIALBOX_ROOT_ENV} CACHE "Serialbox install path.")
endif()

if(NOT(DEFINED SERIALBOX_ROOT))
  find_path(SERIALBOX_ROOT NAMES include/serialbox/core/Config.h)
else()
  get_filename_component(_SERIALBOX_ROOT_ABSOLUTE ${SERIALBOX_ROOT} ABSOLUTE)
  set(SERIALBOX_ROOT ${_SERIALBOX_ROOT_ABSOLUTE} CACHE PATH "Serialbox install path.")
endif()

#
# Default version is 2.0.0
#
if(NOT Serialbox_FIND_VERSION)
  if(NOT Serialbox_FIND_VERSION_MAJOR)
    set(Serialbox_FIND_VERSION_MAJOR 2)
  endif(NOT Serialbox_FIND_VERSION_MAJOR)
  if(NOT Serialbox_FIND_VERSION_MINOR)
    set(Serialbox_FIND_VERSION_MINOR 0)
  endif(NOT Serialbox_FIND_VERSION_MINOR)
  if(NOT Serialbox_FIND_VERSION_PATCH)
    set(Serialbox_FIND_VERSION_PATCH 0)
  endif(NOT Serialbox_FIND_VERSION_PATCH)
  set(Serialbox_FIND_VERSION 
  "${Serialbox_FIND_VERSION_MAJOR}.${Serialbox_FIND_VERSION_MINOR}.${Serialbox_FIND_VERSION_PATCH}")
endif(NOT Serialbox_FIND_VERSION)

#===---------------------------------------------------------------------------------------------===
#   Find serialbox headers
#====--------------------------------------------------------------------------------------------===
if(SERIALBOX_ROOT)
  find_path(SERIALBOX_INCLUDE_DIRS NAMES serialbox/core/Config.h HINTS ${SERIALBOX_ROOT}/include)
endif()

#===---------------------------------------------------------------------------------------------===
#   Read config file (serialbox/core/Config.h)
#====--------------------------------------------------------------------------------------------===
if(SERIALBOX_ROOT)
  file(READ ${SERIALBOX_ROOT}/include/serialbox/core/Config.h _CONFIG_FILE)

  # Get version  
  string(REGEX MATCH "define[ \t]+SERIALBOX_VERSION_MAJOR[ \t]+([0-9]+)" _MAJOR "${_CONFIG_FILE}")
  set(SERIALBOX_MAJOR_VERSION "${CMAKE_MATCH_1}")
  string(REGEX MATCH "define[ \t]+SERIALBOX_VERSION_MINOR[ \t]+([0-9]+)" _MINOR "${_CONFIG_FILE}")
  set(SERIALBOX_MINOR_VERSION "${CMAKE_MATCH_1}")
  string(REGEX MATCH "define[ \t]+SERIALBOX_VERSION_PATCH[ \t]+([0-9]+)" _PATCH "${_CONFIG_FILE}")
  set(SERIALBOX_PATCH_VERSION "${CMAKE_MATCH_1}")

  set(SERIALBOX_VERSION 
      "${SERIALBOX_MAJOR_VERSION}.${SERIALBOX_MINOR_VERSION}.${SERIALBOX_PATCH_VERSION}")

  # Get Boost version
  #
  #   BOOST_VERSION % 100 is the patch level
  #   BOOST_VERSION / 100 % 1000 is the minor version
  #   BOOST_VERSION / 100000 is the major version
  #
  string(REGEX MATCH "define[ \t]+SERIALBOX_BOOST_VERSION[ \t]+([0-9]+)" _BOOST "${_CONFIG_FILE}")
  set(_BOOST_VERSION "${CMAKE_MATCH_1}")
  math(EXPR _BOOST_MAJOR_VERSION "${_BOOST_VERSION} / 100000")
  math(EXPR _BOOST_MINOR_VERSION "${_BOOST_VERSION} / 100 % 1000")
  set(SERIALBOX_BOOST_VERSION "${_BOOST_MAJOR_VERSION}.${_BOOST_MINOR_VERSION}")

  # Check for OpenSSL support
  string(REGEX MATCH "define[ \t]+SERIALBOX_HAS_OPENSSL[ \t]+([0-9]+)" _OPENSSL "${_CONFIG_FILE}")
  if(CMAKE_MATCH_1)
    set(SERIALBOX_HAS_OPENSSL TRUE)
  else()
    set(SERIALBOX_HAS_OPENSSL FALSE)
  endif()
  
  # Check for NetCDF support
  string(REGEX MATCH "define[ \t]+SERIALBOX_HAS_NETCDF[ \t]+([0-9]+)" _NETCDF "${_CONFIG_FILE}")
  if(CMAKE_MATCH_1)
    set(SERIALBOX_HAS_NETCDF TRUE)
  else()
    set(SERIALBOX_HAS_NETCDF FALSE)
  endif()
  
  # Check if logging is ON
  string(REGEX MATCH "define[ \t]+SERIALBOX_HAS_LOGGING[ \t]+([0-9]+)" _LOGGING "${_CONFIG_FILE}")
  set(SERIALBOX_HAS_LOGGING "${CMAKE_MATCH_1}")
endif()

#===---------------------------------------------------------------------------------------------===
#   Find Serialbox libraries
#====--------------------------------------------------------------------------------------------===
if(SERIALBOX_ROOT)

  if(SERIALBOX_USE_SHARED_LIBS)
    find_library(SERIALBOX_CXX_LIBRARIES 
                 NAMES "libSerialboxCore${CMAKE_SHARED_LIBRARY_SUFFIX}" 
                 HINTS ${SERIALBOX_ROOT}/lib)
    find_library(SERIALBOX_C_LIBRARIES 
                 NAMES "libSerialboxC${CMAKE_SHARED_LIBRARY_SUFFIX}"
                 HINTS ${SERIALBOX_ROOT}/lib)
    find_library(SERIALBOX_FORTRAN_LIBRARIES 
                 NAMES "libSerialboxFortran${CMAKE_SHARED_LIBRARY_SUFFIX}" 
                 HINTS ${SERIALBOX_ROOT}/lib)     
  else()
    find_library(SERIALBOX_CXX_LIBRARIES 
                 NAMES libSerialboxCore.a 
                 HINTS ${SERIALBOX_ROOT}/lib)
    find_library(SERIALBOX_C_LIBRARIES 
                 NAMES libSerialboxC.a 
                 HINTS ${SERIALBOX_ROOT}/lib)
    find_library(SERIALBOX_FORTRAN_LIBRARIES 
                 NAMES libSerialboxFortran.a 
                 HINTS ${SERIALBOX_ROOT}/lib)     
  endif()
  
  if(SERIALBOX_C_LIBRARIES)
    set(SERIALBOX_HAS_C TRUE)
  endif()
  
  if(SERIALBOX_FORTRAN_LIBRARIES)
    set(SERIALBOX_HAS_FORTRAN TRUE)
  endif()
  
  #
  # Check if requested languages are found
  #
  set(SERIALBOX_LANGUAGE_OK TRUE)
  list(FIND Serialbox_FIND_COMPONENTS C _C_REQUESTED)
  list(FIND Serialbox_FIND_COMPONENTS Fortran _Fortran_REQUESTED)
  
  if(NOT(_C_REQUESTED EQUAL -1) AND NOT(SERIALBOX_HAS_C))
    set(SERIALBOX_LANGUAGE_OK FALSE)    
  endif()
  
  if(NOT(_Fortran_REQUESTED EQUAL -1) AND NOT(SERIALBOX_HAS_FORTRAN))
    set(SERIALBOX_LANGUAGE_OK FALSE)    
  endif()
  
  set(SERIALBOX_LIBRARY_DIR ${SERIALBOX_ROOT}/lib)
endif()

#===---------------------------------------------------------------------------------------------===
#   Find external libraries
#====--------------------------------------------------------------------------------------------===
if(SERIALBOX_ROOT AND NOT(DEFINED SERIALBOX_NO_EXTERNAL_LIBS))

  set(SERIALBOX_EXTERNAL_LIBRARIES)

  #
  # Pthreads
  #
  find_package(Threads REQUIRED QUIET)
  if(CMAKE_THREAD_LIBS_INIT)
    list(APPEND SERIALBOX_EXTERNAL_LIBRARIES ${CMAKE_THREAD_LIBS_INIT})
  endif()

  #
  # Boost (Serialbox always uses the shared Boost libraries)
  #
  set(Boost_USE_STATIC_LIBS OFF)
  set(Boost_USE_STATIC_RUNTIME OFF)
  set(Boost_USE_MULTITHREADED ON)  
  
  set(_REQUIRED_BOOST_COMPONENTS filesystem system)
  if(SERIALBOX_HAS_LOGGING)
    list(APPEND _REQUIRED_BOOST_COMPONENTS log)
  endif()
  
  find_package(Boost 
               ${SERIALBOX_BOOST_VERSION} EXACT COMPONENTS ${_REQUIRED_BOOST_COMPONENTS})
  if(Boost_FOUND)
    list(APPEND SERIALBOX_INCLUDE_DIRS ${Boost_INCLUDE_DIRS})
    list(APPEND SERIALBOX_EXTERNAL_LIBRARIES ${Boost_LIBRARIES})
  else()
  
    # Give some diagnostic infos
    set(WARN_STR "Serialbox: Boost (${SERIALBOX_BOOST_VERSION}) NOT found!")
  
    if(DEFINED Boost_LIB_VERSION)
      string(REPLACE "_" "." FOUND_BOOST_VERSION ${Boost_LIB_VERSION})
      list(APPEND WARN_STR " (Found Boost ${FOUND_BOOST_VERSION})")  
    endif()
    
    list(APPEND WARN_STR "\nRequired components:")
    
    foreach(component ${_REQUIRED_BOOST_COMPONENTS})
      list(APPEND WARN_STR "\n - ${component}")
    endforeach()
    
    message(WARNING ${WARN_STR} "\n")
  endif()
  
  #
  # OpenSSL
  #
  if(SERIALBOX_HAS_OPENSSL)
    find_package(OpenSSL QUIET)
    if(OpenSSL_FOUND)  
      list(APPEND SERIALBOX_EXTERNAL_LIBRARIES ${OPENSSL_LIBRARIES})
    else()
      message(WARNING "Serialbox depends on the OpenSSL libraries")
    endif()
  endif()
  
  #
  # NetCDF
  #
  if(SERIALBOX_HAS_NETCDF)

    set(NETCDF_ROOT_ENV $ENV{NETCDF_ROOT})
    if(NETCDF_ROOT_ENV)
      set(NETCDF_ROOT ${NETCDF_ROOT_ENV} CACHE "NetCDF install path.")
    endif()
    
    if(NOT(DEFINED NETCDF_ROOT))
      find_path(NETCDF_ROOT NAMES include/netcdf.h)
    else()
      get_filename_component(_NETCDF_ROOT_ABSOLUTE ${NETCDF_ROOT} ABSOLUTE)
      set(NETCDF_ROOT ${_NETCDF_ROOT_ABSOLUTE} CACHE PATH "NetCDF-4 install path.")
    endif()

    find_library(NETCDF_LIBRARIES NAMES netcdf HINTS ${NETCDF_ROOT}/lib)
    if(NETCDF_LIBRARIES)
      mark_as_advanced(NETCDF_LIBRARIES)
      list(APPEND SERIALBOX_EXTERNAL_LIBRARIES ${NETCDF_LIBRARIES})
    else()
      message(WARNING 
              "Serialbox depends on the NetCDF-4 libraries. (Try setting NETCDF_ROOT in the env)")
    endif()
  endif()

  #
  # Only append if library was found (otherwise we confuse find_package_handle_standard_args)
  #
  if(SERIALBOX_CXX_LIBRARIES)
    list(APPEND SERIALBOX_CXX_LIBRARIES ${SERIALBOX_EXTERNAL_LIBRARIES})
  endif()
  
  if(SERIALBOX_C_LIBRARIES)
    list(APPEND SERIALBOX_C_LIBRARIES ${SERIALBOX_EXTERNAL_LIBRARIES})
  endif()
  
  if(SERIALBOX_FORTRAN_LIBRARIES)
    list(APPEND SERIALBOX_FORTRAN_LIBRARIES ${SERIALBOX_EXTERNAL_LIBRARIES})
  endif()
endif()

#===---------------------------------------------------------------------------------------------===
#   Find pp_ser.py
#====--------------------------------------------------------------------------------------------===
if(SERIALBOX_ROOT)
  find_file(SERIALBOX_PPSER pp_ser.py ${SERIALBOX_ROOT}/python/pp_ser)
endif()

#===---------------------------------------------------------------------------------------------===
# Report result 
#====--------------------------------------------------------------------------------------------===
find_package_handle_standard_args(
  Serialbox
  FAIL_MESSAGE "Could NOT find Serialbox. (Try setting SERIALBOX_ROOT in the env)"
  REQUIRED_VARS SERIALBOX_ROOT 
                SERIALBOX_INCLUDE_DIRS
                SERIALBOX_CXX_LIBRARIES
                SERIALBOX_LANGUAGE_OK
                SERIALBOX_LIBRARY_DIR
  VERSION_VAR SERIALBOX_VERSION)

mark_as_advanced(SERIALBOX_INCLUDE_DIRS
                 SERIALBOX_HAS_C
                 SERIALBOX_HAS_FORTRAN
                 SERIALBOX_CXX_LIBRARIES
                 SERIALBOX_LIBRARY_DIR
                 SERIALBOX_C_LIBRARIES
                 SERIALBOX_FORTRAN_LIBRARIES
                 SERIALBOX_LANGUAGE_OK
                 SERIALBOX_PPSER
                 SERIALBOX_HAS_NETCDF
                 SERIALBOX_HAS_OPENSSL)

if(SERIALBOX_FOUND)
  message(STATUS "Serialbox version: ${SERIALBOX_VERSION}")

  list(GET SERIALBOX_CXX_LIBRARIES 0 CXX_LIB_PATH)
  get_filename_component(CXX_LIB ${CXX_LIB_PATH} NAME)
  message(STATUS "  ${CXX_LIB}")
  
  if(SERIALBOX_HAS_C)
    list(GET SERIALBOX_C_LIBRARIES 0 C_LIB_PATH)
    get_filename_component(C_LIB ${C_LIB_PATH} NAME)
    message(STATUS "  ${C_LIB}")
  endif()
  
  if(SERIALBOX_HAS_FORTRAN)
    list(GET SERIALBOX_FORTRAN_LIBRARIES 0 FORTRAN_LIB_PATH)
    get_filename_component(FORTRAN_LIB ${FORTRAN_LIB_PATH} NAME)
    message(STATUS "  ${FORTRAN_LIB}")
  endif()

else()
  
  # Give some hints what is missing
  set(ERR_MSG "")
  
  if(IS_DIRECTORY ${SERIALBOX_ROOT} AND SERIALBOX_USE_SHARED_LIBS)
    set(ERR_MSG "${ERR_MSG}\n   - Shared libraries are missing!")
  endif()
  
  if(NOT(_C_REQUESTED EQUAL -1) AND NOT(SERIALBOX_HAS_C))
    set(ERR_MSG "${ERR_MSG}\n   - C interface is missing!")
  endif()

  if(NOT(_Fortran_REQUESTED EQUAL -1) AND NOT(SERIALBOX_HAS_FORTRAN))
    set(ERR_MSG "${ERR_MSG}\n   - Fortran interface is missing!")
  endif()
  
  if(${Serialbox_FIND_REQUIRED})
    message(FATAL_ERROR "${ERR_MSG}\n")
  else()
    message("${ERR_MSG}\n")
  endif()
endif()

