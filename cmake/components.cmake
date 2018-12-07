set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_PREFIX}/lib/cmake/gridtools)
### This file define the GridTools components and their dependencies

### The following function works like this:
###
### First argument the name of the library to build
###
### All other arguments are the sources for compiling the library. If this list is empty this will be a header-only library!
###
function(generate_target_for)
    cmake_parse_arguments(TARGET "" "NAME" "SOURCES" ${ARGN})
    if ("${TARGET_SOURCES} " STREQUAL " ")
        add_library(${TARGET_NAME} INTERFACE)
        target_include_directories(${TARGET_NAME} INTERFACE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/> $<INSTALL_INTERFACE:include/> )

    else ()
        add_library(${TARGET_NAME} ${TARGET_SOURCES})
        target_include_directories(${TARGET_NAME}
                     PUBLIC
                        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/>
                        $<INSTALL_INTERFACE:include>
                     PRIVATE
                        ${CMAKE_SOURCE_DIR}/src
        )
    endif()
endfunction(generate_target_for)

include (CMakeDependentOption)
set(INSTALL_ALL ON CACHE BOOL "Install all")
CMAKE_DEPENDENT_OPTION(INSTALL_GCL "GCL/communication component" OFF "NOT INSTALL_ALL" ON)
CMAKE_DEPENDENT_OPTION(INSTALL_BOUNDARY_CONDITIONS "Boundary conditions component" OFF "NOT INSTALL_ALL" ON)
CMAKE_DEPENDENT_OPTION(INSTALL_DISTRIBUTED_BOUNDARIES "Distributed boundaries component" OFF "NOT INSTALL_ALL" ON)
CMAKE_DEPENDENT_OPTION(INSTALL_COMMON "Common component" OFF "NOT INSTALL_ALL" ON)
CMAKE_DEPENDENT_OPTION(INSTALL_STENCIL_COMPOSITION "Stencil composition component" OFF "NOT INSTALL_ALL" ON)
CMAKE_DEPENDENT_OPTION(INSTALL_STORAGE "Storage component" OFF "NOT INSTALL_ALL" ON)
CMAKE_DEPENDENT_OPTION(INSTALL_C_BINDINGS "C/Fortran bindings component" OFF "NOT INSTALL_ALL" ON)
CMAKE_DEPENDENT_OPTION(INSTALL_INTERFACE "Interface component" OFF "NOT INSTALL_ALL" ON)
CMAKE_DEPENDENT_OPTION(INSTALL_TOOLS "Tools component" OFF "NOT INSTALL_ALL" ON)

macro( turn_on comp )
       if ( ${comp} )
       else()
          message("--> Component ${comp} is needed")
          set( ${comp} ON )
       endif()
endmacro( turn_on )

macro( define_component name )
       message("Defining component ${name}")
       turn_on( ${name} )
       if ("${ARGN} " STREQUAL " ")
       else()
           foreach( comp ${ARGN} )
               turn_on( ${comp} )
           endforeach()
       endif()
endmacro( define_component )

if( INSTALL_COMMON )
define_component( COMPONENT_COMMON )
endif()

if( INSTALL_C_BINDINGS )
define_component( COMPONENT_C_BINDINGS COMPONENT_COMMON )
endif()

if( INSTALL_GCL )
define_component( COMPONENT_GCL COMPONENT_COMMON )
endif()

if( INSTALL_BOUNDARY_CONDITIONS )
define_component( COMPONENT_BOUNDARY_CONDITIONS COMPONENT_COMMON )
endif()

if( INSTALL_DISTRIBUTED_BOUNDARIES )
define_component( COMPONENT_DISTRIBUTED_BOUNDARIES COMPONENT_GCL COMPONENT_BOUNDARY_CONDITIONS COMPONENT_COMMON)
endif()

if( INSTALL_STENCIL_COMPOSITION )
define_component( COMPONENT_STENCIL_COMPOSITION COMPONENT_COMMON )
endif()

if( INSTALL_STORAGE )
define_component( COMPONENT_STORAGE COMPONENT_COMMON )
endif()

if( INSTALL_INTERFACE )
define_component( COMPONENT_INTERFACE COMPONENT_C_BINDINGS COMPONENT_COMMON )
endif()

if( INSTALL_TOOLS )
define_component( COMPONENT_TOOLS COMPONENT_COMMON )
endif()

if( INSTALL_ALL )
define_component( COMPONENT_ALLS COMPONENT_GCL COMPONENT_BOUNDARY_CONDITIONS COMPONENT_DISTRIBUTED_BOUNDARIES COMPONENT_COMMON
    COMPONENT_STENCIL_COMPOSITION COMPONENT_STORAGE COMPONENT_C_BINDINGS COMPONENT_INTERFACE COMPONENT_TOOLS)
endif()
