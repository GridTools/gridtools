
##!  Copyright (c) 2012, MAURO BIANCO, UGO VARETTO, SWISS NATIONAL SUPERCOMPUTING CENTRE (CSCS)
##!  All rights reserved.

##!  Redistribution and use in source and binary forms, with or without
##!  modification, are permitted provided that the following conditions are met:
##!      * Redistributions of source code must retain the above copyright
##!        notice, this list of conditions and the following disclaimer.
##!      * Redistributions in binary form must reproduce the above copyright
##!        notice, this list of conditions and the following disclaimer in the
##!        documentation and/or other materials provided with the distribution.
##!      * Neither the name of the Swiss National Supercomputing Centre (CSCS) nor the
##!        names of its contributors may be used to endorse or promote products
##!        derived from this software without specific prior written permission.

##!  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
##!  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
##!  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
##!  DISCLAIMED. IN NO EVENT SHALL MAURO BIANCO, UGO VARETTO, OR 
##!  SWISS NATIONAL SUPERCOMPUTING CENTRE (CSCS), BE LIABLE FOR ANY
##!  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
##!  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
##!  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
##!  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
##!  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
##!  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



Cmake_minimum_required( VERSION 2.8 )

project(StencilLibrary)

#Boost
find_package( Boost )
if( Boost_FOUND )
  include_directories( ${Boost_INCLUDE_DIR} )
endif()


#MPI
set( USE_APRUN "ON" CACHE BOOL "Use uprun on rosa?" )
set( GCL_MPI "ON" CACHE BOOL "Compile MPI version" )
set( GCL_GPU "OFF" CACHE BOOL "Compile with GPU support (CUDA)" )
set( GCL_ONLY "ON" CACHE BOOL "If on only library is build but not the examples and tests" )
set( USE_MPI_COMPILER "OFF" CACHE BOOL "On rosa turn this flag off since compiler takes care of mpi already" )
set( HOST_SPECIFIC_OPTIONS "" CACHE STRING "Options passed only to HOST COMPILER and not ACCELERATOR COMPILER" )

if( "${USE_APRUN}" STREQUAL "ON" )
    set(RUN_CMD "/usr/bin/aprun -n 1")
    set(MPI_RUN_CMD "/usr/bin/aprun -n 4")
else()
    set(RUN_CMD "")
    set(MPI_RUN_CMD "mpirun -n 4")
endif()


if( "${GCL_GPU}" STREQUAL "ON" )
  include(FindCUDA)
  include_directories(${CUDA_INCLUDE_DIRS})
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GCL_GPU_")
else()
set (CUDA_LIBRARIES "")
endif()


if( "${GCL_MPI}" STREQUAL "ON" )
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GCL_MPI_")
  if( "${USE_MPI_COMPILER}" STREQUAL "ON" )
    FIND_PACKAGE(MPI REQUIRED)
    INCLUDE(CMakeForceCompiler)
    CMAKE_FORCE_CXX_COMPILER(mpicxx "MPI C++ Compiler")
  endif()
endif()


include_directories( L3/include/ ) ## Library header files
include_directories( L3/src/ ) ## Library source files
include_directories( L2/include/ ) ## Library header files
include_directories( L2/src/ ) ## Library source files
include_directories( . ) ## Library header files

### Library
ADD_LIBRARY( gcl L3/src/GCL.cpp L3/include/GCL.h L2/include/stats_collector.h)

set(EXECUTABLE_OUTPUT_PATH "build")
set(LIBRARY_OUTPUT_PATH "lib")

set( OLD_OPTIONS "${CMAKE_CXX_FLAGS}" )


### Executables
if("${GCL_ONLY}" STREQUAL "OFF")
###   message("${HOST_SPECIFIC_OPTIONS}")
###   message("${CMAKE_CXX_FLAGS}")
###   set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${HOST_SPECIFIC_OPTIONS}" )
###   message("${CMAKE_CXX_FLAGS}")
  ### L3
  if( "${GCL_MPI}" STREQUAL "ON" )
    #CPU 2D MPI tests
    foreach(srcfile test_generic_all_to_all Halo_Exchange_test Halo_Exchange_test_3D Halo_Exchange_test_2  proc_grid_2D_test proc_grid_3D_test)
      add_executable( ${srcfile} L3/test/${srcfile}.cpp)
      add_dependencies(${srcfile} gcl)
          set_target_properties(${srcfile} PROPERTIES COMPILE_FLAGS "${HOST_SPECIFIC_OPTIONS}")
      target_link_libraries(${srcfile} gcl ${MPI_ADDITIONAL_LIBS} ${CUDA_LIBRARIES})
    endforeach(srcfile)
  endif()

  ### L2
  if( "${GCL_MPI}" STREQUAL "ON" )
    #CPU 2D MPI tests
    foreach(srcfile test_halo_exchange test_halo_exchange_2D test_halo_exchange_3D test_halo_exchange_3D_all test_halo_exchange_3D_all_2 test_halo_exchange_3D_all_3 test_halo_exchange_3D_generic test_async_progress_3D_generic test_halo_exchange_2D_generic test_halo_exchange_3D_generic_full test_all_to_all_halo test_all_to_all_ij_ji test_all_to_all_halo_3D test_descriptors test_descriptors_2D big_test test_descriptors_3D test_empty_descriptors_2D test_empty_descriptors_3D)
      add_executable( ${srcfile} L2/test/${srcfile}.cpp)
      add_dependencies(${srcfile} gcl)
      set_target_properties(${srcfile} PROPERTIES COMPILE_FLAGS "${HOST_SPECIFIC_OPTIONS}")
      target_link_libraries(${srcfile} gcl ${MPI_ADDITIONAL_LIBS} ${CUDA_LIBRARIES})
    endforeach(srcfile)
  endif()

  foreach(srcfile boolist ndloops layout_map)
    add_executable( ${srcfile} utils/test/${srcfile}.cpp)
    set_target_properties(${srcfile} PROPERTIES COMPILE_FLAGS "${HOST_SPECIFIC_OPTIONS}")
  endforeach(srcfile)


  if( "${GCL_GPU}" STREQUAL "ON" )
    ##set(CMAKE_CXX_FLAGS "${OLD_OPTIONS}")
    if( "${GCL_MPI}" STREQUAL "ON" )
      if ( "${USE_MPI_COMPILER}" )
        #message( "MPI enabled for CUDA - compiling with mpi compiler" )
        include_directories( ${MPI_INCLUDE_PATH} )
        #message( "MPI include path: ${MPI_INCLUDE_PATH}" )
        #link_directories( ${MPI_LIBRARY} )
      else( "${USE_MPI_COMPILER}" )
        #message( "MPI enabled for CUDA - compiling with wrappers" )
        set( MPI_INCLUDE_PATH $ENV{MPICH_DIR}/include )
        #set( MPI_LIBRARY_PATH $ENV{MPICH_DIR}/lib )
        #set( CUDA_MPI_LIB mpi )
        #link_directories( ${MPI_LIBRARY_PATH} )
        include_directories( ${MPI_INCLUDE_PATH} )
        #message( "MPI include path: ${MPI_INCLUDE_PATH}" )
      endif( "${USE_MPI_COMPILER}" )      
    endif()
    if( "${GCL_MPI}" STREQUAL "ON" )
      if(NOT MSVC)
        cuda_add_executable( cuda_test_halo_exchange_3D_all L2/test/test_halo_exchange_3D_all.cu )
        add_dependencies( cuda_test_halo_exchange_3D_all gcl ) 
        target_link_libraries( cuda_test_halo_exchange_3D_all gcl ${CUDA_MPI_LIB} )

        cuda_add_executable( cuda_test_halo_exchange_3D_all_2 L2/test/test_halo_exchange_3D_all_2.cu )
        add_dependencies( cuda_test_halo_exchange_3D_all_2 gcl ) 
        target_link_libraries( cuda_test_halo_exchange_3D_all_2 gcl ${CUDA_MPI_LIB} )

        cuda_add_executable( cuda_test_halo_exchange_3D_all_3 L2/test/test_halo_exchange_3D_all_3.cu )
        add_dependencies( cuda_test_halo_exchange_3D_all_3 gcl ) 
        target_link_libraries( cuda_test_halo_exchange_3D_all_3 gcl ${CUDA_MPI_LIB} )

        cuda_add_executable( cuda_test_halo_exchange_3D_generic L2/test/test_halo_exchange_3D_generic.cu )
        add_dependencies( cuda_test_halo_exchange_3D_generic gcl ) 
        target_link_libraries( cuda_test_halo_exchange_3D_generic gcl ${CUDA_MPI_LIB} )

        cuda_add_executable( cuda_test_async_progress_3D_generic L2/test/test_async_progress_3D_generic.cu )
        add_dependencies( cuda_test_async_progress_3D_generic gcl ) 
        target_link_libraries( cuda_test_async_progress_3D_generic gcl ${CUDA_MPI_LIB} )

        cuda_add_executable( cuda_test_halo_exchange_3D_generic_full L2/test/test_halo_exchange_3D_generic_full.cu )
        add_dependencies( cuda_test_halo_exchange_3D_generic_full gcl ) 
        target_link_libraries( cuda_test_halo_exchange_3D_generic_full gcl ${CUDA_MPI_LIB} )

        cuda_add_executable( cuda_test_halo_exchange_2D_generic L2/test/test_halo_exchange_2D_generic.cu )
        add_dependencies( cuda_test_halo_exchange_2D_generic gcl ) 
        target_link_libraries( cuda_test_halo_exchange_2D_generic gcl ${CUDA_MPI_LIB} )

      endif()
    endif()      
  endif()
endif()

# Make documentation
find_package(Doxygen)
if(DOXYGEN_FOUND)

   if(EXISTS "Doxyfile")
    set(DOXY_CONFIG "Doxyfile")
   endif(EXISTS "Doxyfile")

   add_custom_target(doc ${DOXYGEN_EXECUTABLE} ${DOXY_CONFIG})
endif(DOXYGEN_FOUND)

