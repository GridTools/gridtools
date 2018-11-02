#TODO enable MC
if (ENABLE_X86)
    if(SINGLE_PRECISION)
        add_bindings_library(implementation_float SOURCES implementation.cpp FORTRAN_MODULE_NAME implementation)
        target_compile_definitions(implementation_float PUBLIC ${X86_BACKEND_DEFINE})
        add_bindings_library(implementation_wrapper_float SOURCES implementation-wrapper.cpp FORTRAN_MODULE_NAME implementation_wrapper)
        target_compile_definitions(implementation_wrapper_float PUBLIC ${X86_BACKEND_DEFINE})
    else()
        add_bindings_library(implementation_double SOURCES implementation.cpp FORTRAN_MODULE_NAME implementation)
        target_compile_definitions(implementation_double PUBLIC ${X86_BACKEND_DEFINE})
        add_bindings_library(implementation_wrapper_double SOURCES implementation-wrapper.cpp FORTRAN_MODULE_NAME implementation_wrapper)
        target_compile_definitions(implementation_wrapper_double PUBLIC ${X86_BACKEND_DEFINE})
    endif()

    # the convention is that the user has to compile handle and array_descriptor
    add_library(c_bindings_handle ${CMAKE_SOURCE_DIR}/src/c_bindings/handle.cpp)
    add_library(fortran_bindings ${CMAKE_SOURCE_DIR}/src/c_bindings/array_descriptor.f90 ${CMAKE_SOURCE_DIR}/src/c_bindings/handle.f90)
    target_link_libraries(fortran_bindings c_bindings_handle)

    add_executable(driver driver.c)
    target_link_libraries(driver implementation_c)
    target_link_libraries(driver c_bindings_handle)

    set (CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -cpp")


    add_executable(fdriver fdriver.f90)
    if(SINGLE_PRECISION)
        target_link_libraries(fdriver implementation_float_fortran)
    else()
        target_link_libraries(fdriver implementation_double_fortran)
    endif() 
    target_link_libraries(fdriver fortran_bindings)
    set_target_properties(fdriver PROPERTIES LINKER_LANGUAGE Fortran)

    add_executable(fdriver_wrapper fdriver-wrapper.f90)
    if(SINGLE_PRECISION)
        target_link_libraries(fdriver_wrapper implementation_wrapper_float_fortran)
    else()
        target_link_libraries(fdriver_wrapper implementation_wrapper_double_fortran)
    endif()
    target_link_libraries(fdriver_wrapper fortran_bindings)
    set_target_properties(fdriver_wrapper PROPERTIES LINKER_LANGUAGE Fortran)

endif(ENABLE_X86)
