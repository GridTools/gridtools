message("Downloading gridtools_experimental...")

set(GRIDTOOLS_EXPERIMENTAL_DIR "${CMAKE_SOURCE_DIR}/experimental")

file(MAKE_DIRECTORY ${GRIDTOOLS_EXPERIMENTAL_DIR})

configure_file(${CMAKE_CURRENT_LIST_DIR}/CMakeLists.txt.in ${CMAKE_BINARY_DIR}/download_gridtools_experimental/CMakeLists.txt)

execute_process(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" ${CMAKE_BINARY_DIR}/download_gridtools_experimental/CMakeLists.txt
WORKING_DIRECTORY "${GRIDTOOLS_EXPERIMENTAL_DIR}" )

execute_process(COMMAND "${CMAKE_COMMAND}" --build ${CMAKE_BINARY_DIR}/download_gridtools_experimental
WORKING_DIRECTORY "${GRIDTOOLS_EXPERIMENTAL_DIR}" )

add_subdirectory(${GRIDTOOLS_EXPERIMENTAL_DIR})

message("Downloading gridtools_experimental done!")
