include(bindings)

if (ENABLE_HOST)
    add_bindings_library(implementation implementation.cpp)
    add_bindings_library(implementation_wrapper implementation-wrapper.cpp)
    target_compile_definitions(implementation PUBLIC ${HOST_BACKEND_DEFINE})
    target_compile_definitions(implementation_wrapper PUBLIC ${HOST_BACKEND_DEFINE})

    add_executable(driver driver.c)
    target_link_libraries(driver implementation_c)

    enable_language (Fortran)

    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -cpp")

    add_executable(fdriver fdriver.f90)
    target_link_libraries(fdriver implementation_fortran)

    add_executable(fdriver_wrapper fdriver-wrapper.f90)
    target_link_libraries(fdriver_wrapper implementation_wrapper_fortran)

    # On tave Intel Fortran is not working
    if(CMAKE_Fortran_COMPILER_ID MATCHES "Intel")
        set_target_properties(fdriver PROPERTIES EXCLUDE_FROM_ALL 1)
        set_target_properties(fdriver_wrapper PROPERTIES EXCLUDE_FROM_ALL 1)
    endif()
endif(ENABLE_HOST)
