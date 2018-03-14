
if( NOT ATLAS_FOUND )

  find_path( ATLAS_INCLUDE_DIR 
             NAMES "atlas/mesh.h"
             PATHS 
                ${CMAKE_INSTALL_PREFIX}
                ${ATLAS_PATH}
                ENV ATLAS_PATH 
            PATH_SUFFIXES include
  )

  find_library( ATLAS_LIB
            NAME "libatlas.so"
            PATHS
               ${CMAKE_INSTALL_PREFIX}
                ${ATLAS_PATH}
                ENV ATLAS_PATH
            PATH_SUFFIXES lib
  )

  find_library( ECKIT_LIB
            NAME "libeckit.so"
            PATHS
               ${CMAKE_INSTALL_PREFIX}
                ${ATLAS_PATH} ${ECKIT_PATH}
                ENV ATLAS_PATH
            PATH_SUFFIXES lib
  )

  include(FindPackageHandleStandardArgs)

  # handle the QUIETLY and REQUIRED arguments and set ATLAS_FOUND to TRUE
  find_package_handle_standard_args( ATLAS DEFAULT_MSG
                                     ATLAS_INCLUDE_DIR ATLAS_LIB)

  set(ATLAS_LIBRARIES ${ATLAS_LIB} ${ECKIT_LIB})
  mark_as_advanced( ATLAS_INCLUDE_DIR ATLAS_LIBRARIES ATLAS_LIB)

endif()

