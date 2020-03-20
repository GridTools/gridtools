# for install tree
set(GRIDTOOLS_MODULE_PATH lib/cmake)

include(CMakePackageConfigHelpers)
set(GRIDTOOLS_INCLUDE_PATH include)
configure_package_config_file(cmake/internal/GridToolsConfig.cmake.in
  ${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/install/GridToolsConfig.cmake
  PATH_VARS GRIDTOOLS_MODULE_PATH GRIDTOOLS_INCLUDE_PATH
  INSTALL_DESTINATION lib/cmake)

write_basic_package_version_file(
  ${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/install/GridToolsConfigVersion.cmake
  COMPATIBILITY SameMajorVersion )

install(DIRECTORY include/gridtools/ DESTINATION include/gridtools)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/cmake/public/ DESTINATION "lib/cmake/${PROJECT_NAME}")

# # for build tree
# set(GRIDTOOLS_MODULE_PATH ${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/build-install/lib/cmake)
# set(GRIDTOOLS_INCLUDE_PATH ${PROJECT_SOURCE_DIR}/include)
# configure_package_config_file(cmake/internal/GridToolsConfig.cmake.in
#   ${PROJECT_BINARY_DIR}/GridToolsConfig.cmake
#   PATH_VARS GRIDTOOLS_MODULE_PATH GRIDTOOLS_INCLUDE_PATH
#   INSTALL_DESTINATION ${PROJECT_BINARY_DIR})
# write_basic_package_version_file(
#   ${PROJECT_BINARY_DIR}/GridToolsConfigVersion.cmake
#   COMPATIBILITY SameMajorVersion )

# Install the GridToolsConfig.cmake and GridToolsConfigVersion.cmake
install(FILES "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/install/GridToolsConfig.cmake"
    "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/install/GridToolsConfigVersion.cmake"
  DESTINATION "lib/cmake/${PROJECT_NAME}")


# file(COPY ${PROJECT_SOURCE_DIR}/cmake/public/ DESTINATION "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/build-install/lib/cmake")
