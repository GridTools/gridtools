add_library(implementation implementation.cpp)
target_link_libraries(implementation c_bindings_generator)

add_executable(header_generator header_generator.cpp)
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    target_link_libraries(header_generator -force_load implementation)
else()
    target_link_libraries(header_generator -Wl,-whole-archive implementation)
endif()

add_custom_command(OUTPUT implementation.h
        COMMAND header_generator > implementation.h
        DEPENDS $<TARGET_FILE:header_generator>)

add_custom_target(implementation_header DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/implementation.h)

add_executable(driver driver.c ${CMAKE_CURRENT_BINARY_DIR}/implementation.h)
target_include_directories(driver PUBLIC "${CMAKE_CURRENT_BINARY_DIR}")
target_link_libraries(driver implementation)

add_dependencies(driver implementation_header)
