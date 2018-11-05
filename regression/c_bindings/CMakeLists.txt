if (ENABLE_X86)
    if(SINGLE_PRECISION)
        set(prec float)
    else()
        set(prec double)
    endif()

    add_bindings_library(implementation_${prec} SOURCES implementation.cpp FORTRAN_MODULE_NAME implementation)
    target_compile_options(implementation_${prec} PRIVATE ${GT_CXX_BUILDING_FLAGS} ${GT_CXX_OPTIMIZATION_FLAGS} ${GT_CXX_MANDATORY_FLAGS} -D${X86_BACKEND_DEFINE})
    add_bindings_library(implementation_wrapper_${prec} SOURCES implementation-wrapper.cpp FORTRAN_MODULE_NAME implementation_wrapper)
    target_compile_options(implementation_wrapper_${prec} PRIVATE ${GT_CXX_BUILDING_FLAGS} ${GT_CXX_OPTIMIZATION_FLAGS} ${GT_CXX_MANDATORY_FLAGS} -D${X86_BACKEND_DEFINE})

    # the convention is that the user has to compile handle and array_descriptor
    add_library(c_bindings_handle ${CMAKE_SOURCE_DIR}/src/c_bindings/handle.cpp)
    target_include_directories(c_bindings_handle PUBLIC ${CMAKE_SOURCE_DIR}/include)
    add_library(fortran_bindings ${CMAKE_SOURCE_DIR}/src/c_bindings/array_descriptor.f90 ${CMAKE_SOURCE_DIR}/src/c_bindings/handle.f90)
    target_link_libraries(fortran_bindings c_bindings_handle)

    add_executable(driver driver.c)
    target_link_libraries(driver implementation_${prec}_c)
    target_link_libraries(driver c_bindings_handle)
    target_compile_options(driver PRIVATE ${GT_C_BUILDING_FLAGS})

    set (CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -cpp")

    add_executable(fdriver fdriver.f90)
    target_link_libraries(fdriver implementation_${prec}_fortran)
    target_link_libraries(fdriver fortran_bindings)
    set_target_properties(fdriver PROPERTIES LINKER_LANGUAGE Fortran)

    add_executable(fdriver_wrapper fdriver-wrapper.f90)
    target_link_libraries(fdriver_wrapper implementation_wrapper_${prec}_fortran)
    target_link_libraries(fdriver_wrapper fortran_bindings)
    set_target_properties(fdriver_wrapper PROPERTIES LINKER_LANGUAGE Fortran)
endif(ENABLE_X86)
