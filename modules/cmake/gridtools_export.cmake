macro( gridtools_export )

  set( _targets ${ARGV} )

  export( TARGETS ${_targets} FILE cmake/${PROJECT_NAME}-exports.cmake NAMESPACE gridtools:: )
  
  install( TARGETS ${_targets} EXPORT ${PROJECT_NAME}-exports DESTINATION share/gridtools/cmake/ )
  install( EXPORT ${PROJECT_NAME}-exports DESTINATION share/gridtools/cmake NAMESPACE gridtools:: )

  foreach( target ${_targets} ) # For build-tree exports
    add_library( gridtools::${target} ALIAS ${target} )
  endforeach()

  configure_file(
    ${PROJECT_NAME}-config.cmake.in
    "${PROJECT_BINARY_DIR}/cmake/${PROJECT_NAME}-config.cmake"
    @ONLY )

  include(CMakePackageConfigHelpers)
  write_basic_package_version_file(
    "${PROJECT_BINARY_DIR}/cmake/${PROJECT_NAME}-config-version.cmake"
    VERSION ${GRIDTOOLS_VERSION}
    COMPATIBILITY AnyNewerVersion
  )
    
  # Install the config and config-version files
  install(
    FILES
      "${PROJECT_BINARY_DIR}/cmake/${PROJECT_NAME}-config.cmake"
      "${PROJECT_BINARY_DIR}/cmake/${PROJECT_NAME}-config-version.cmake"
    DESTINATION share/gridtools/cmake )

  foreach( target ${_targets} )
    string(REPLACE gridtools_ "" target_without_prefix ${target} )
    install( DIRECTORY "${GRIDTOOLS_SOURCE_DIR}/include/gridtools/${target_without_prefix}"
             DESTINATION include/gridtools )
  endforeach()

  # We could export the package for use in build trees
  # WARNING: this registers the build-tree in a global CMake-registry located in ~/.cmake,
  #          which is undesired when having multiple builds
  # export( PACKAGE ${PROJECT_NAME} )

  # Instead, cache the location of the build-tree config files
  set( ${PROJECT_NAME}_DIR ${PROJECT_BINARY_DIR}/cmake CACHE STRING "" )

  # Export variable to notify super-builds that we are in the build-tree
  set( ${UPPERCASE_PROJECT_NAME}_TARGETS_EXPORTED TRUE CACHE STRING "" )

endmacro()
