macro( gridtools_export )

  set( gridtools_targets ${ARGV} )

  foreach( target ${gridtools_targets} )
    string(REPLACE gridtools_ "" target_without_prefix ${target} )
    add_library( gridtools::${target_without_prefix} ALIAS ${target} )
    list( APPEND modules ${target_without_prefix} )
    list( APPEND targets gridtools::${target_without_prefix} )
    set_target_properties( ${target} PROPERTIES EXPORT_NAME ${target_without_prefix} )
  endforeach()

  export( TARGETS ${gridtools_targets} FILE ${PROJECT_NAME}-export.cmake NAMESPACE gridtools:: )
  install( TARGETS ${gridtools_targets} EXPORT ${PROJECT_NAME}-export DESTINATION share/${PROJECT_NAME}/cmake/ )
  install( EXPORT ${PROJECT_NAME}-export DESTINATION share/${PROJECT_NAME}/cmake NAMESPACE gridtools:: )

  configure_file(
    ${PROJECT_NAME}-config.cmake.in
    "${PROJECT_BINARY_DIR}/${PROJECT_NAME}-config.cmake"
    @ONLY )

  include(CMakePackageConfigHelpers)
  write_basic_package_version_file(
    "${PROJECT_BINARY_DIR}/${PROJECT_NAME}-config-version.cmake"
    VERSION ${GRIDTOOLS_VERSION}
    COMPATIBILITY AnyNewerVersion
  )
    
  # Install the config and config-version files
  install(
    FILES
      "${PROJECT_BINARY_DIR}/${PROJECT_NAME}-config.cmake"
      "${PROJECT_BINARY_DIR}/${PROJECT_NAME}-config-version.cmake"
    DESTINATION share/${PROJECT_NAME}/cmake )

  foreach( module ${modules} )
    install( DIRECTORY "${GRIDTOOLS_SOURCE_DIR}/include/gridtools/${module}"
             DESTINATION include/gridtools )
  endforeach()

  # We could export the package for use in build trees
  # WARNING: this registers the build-tree in a global CMake-registry located in ~/.cmake,
  #          which is undesired when having multiple builds
  # export( PACKAGE ${PROJECT_NAME} )

  # Instead, cache the location of the build-tree config files
  set( ${PROJECT_NAME}_DIR ${PROJECT_BINARY_DIR} CACHE STRING "" )

  # Export variable to notify super-builds that we are in the build-tree
  set( ${UPPERCASE_PROJECT_NAME}_TARGETS_EXPORTED TRUE CACHE STRING "" )

endmacro()
