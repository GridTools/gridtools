if (ENABLE_X86)
    add_bindings_library(implementation implementation.cpp)
    add_bindings_library(implementation_wrapper implementation-wrapper.cpp)
    target_compile_definitions(implementation PUBLIC ${X86_BACKEND_DEFINE})
    target_compile_definitions(implementation_wrapper PUBLIC ${X86_BACKEND_DEFINE})

    add_executable(driver driver.c)
    target_link_libraries(driver implementation_c)

    set (CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -cpp")

    add_executable(fdriver fdriver.f90)
    target_link_libraries(fdriver implementation_fortran)
    set_target_properties(fdriver PROPERTIES LINKER_LANGUAGE Fortran)

    add_executable(fdriver_wrapper fdriver-wrapper.f90)
    target_link_libraries(fdriver_wrapper implementation_wrapper_fortran)
    set_target_properties(fdriver_wrapper PROPERTIES LINKER_LANGUAGE Fortran)

endif(ENABLE_X86)
