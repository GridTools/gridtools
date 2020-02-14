# this registers the build-tree with a global CMake-registry
export(PACKAGE GridTools)

# for install tree
set(GRIDTOOLS_MODULE_PATH lib/cmake)

include(CMakePackageConfigHelpers)
set(GRIDTOOLS_INCLUDE_PATH include)
set(GT_CPP_BINDGEN_CONFIG_LOCATION "\${CMAKE_CURRENT_LIST_DIR}")
configure_package_config_file(cmake/GridToolsConfig.cmake.in
  ${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/install/GridToolsConfig.cmake
  PATH_VARS GRIDTOOLS_MODULE_PATH GRIDTOOLS_INCLUDE_PATH GT_CPP_BINDGEN_CONFIG_LOCATION
  INSTALL_DESTINATION lib/cmake)

write_basic_package_version_file(
  ${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/install/GridToolsConfigVersion.cmake
  COMPATIBILITY SameMajorVersion )

# for build tree
set(GRIDTOOLS_MODULE_PATH ${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/build-install/lib/cmake)
set(GRIDTOOLS_INCLUDE_PATH ${PROJECT_SOURCE_DIR}/include)
FetchContent_GetProperties(cpp_bindgen)
set(GT_CPP_BINDGEN_CONFIG_LOCATION ${cpp_bindgen_BINARY_DIR})
configure_package_config_file(cmake/GridToolsConfig.cmake.in
  ${PROJECT_BINARY_DIR}/GridToolsConfig.cmake
  PATH_VARS GRIDTOOLS_MODULE_PATH GRIDTOOLS_INCLUDE_PATH GT_CPP_BINDGEN_CONFIG_LOCATION
  INSTALL_DESTINATION ${PROJECT_BINARY_DIR})
write_basic_package_version_file(
  ${PROJECT_BINARY_DIR}/GridToolsConfigVersion.cmake
  COMPATIBILITY SameMajorVersion )

install(TARGETS cpp_bindgen_interface EXPORT GridToolsTargets)  #TODO remove cpp_bindgen_interface in GT 2.0

export(EXPORT GridToolsTargets
    FILE ${PROJECT_BINARY_DIR}/GridToolsTargets.cmake
    NAMESPACE GridTools::
)

install(EXPORT GridToolsTargets
  FILE GridToolsTargets.cmake
  NAMESPACE GridTools::
  DESTINATION lib/cmake
)

install(DIRECTORY include/gridtools/ DESTINATION include/gridtools)

# Install the GridToolsConfig.cmake and GridToolsConfigVersion.cmake
install(FILES "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/install/GridToolsConfig.cmake"
    "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/install/GridToolsConfigVersion.cmake"
  DESTINATION "lib/cmake")

set(CMAKE_SOURCES
    "${PROJECT_SOURCE_DIR}/cmake/gt_bindings.cmake" # TODO remove in GT 2.0
    "${PROJECT_SOURCE_DIR}/cmake/fortran_helpers.cmake"
    "${PROJECT_SOURCE_DIR}/cmake/workaround_mpi.cmake"
    "${PROJECT_SOURCE_DIR}/cmake/workaround_check_language.cmake"
    )

install(FILES ${CMAKE_SOURCES} DESTINATION "lib/cmake")

file(COPY ${CMAKE_SOURCES} DESTINATION "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/build-install/lib/cmake")
