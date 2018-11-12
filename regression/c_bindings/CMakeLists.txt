if (GT_ENABLE_TARGET_X86)
    if(SINGLE_PRECISION)
        set(prec float)
    else()
        set(prec double)
    endif()

    add_bindings_library(implementation_${prec} SOURCES implementation.cpp FORTRAN_MODULE_NAME implementation)
    target_compile_options(implementation_${prec} PRIVATE ${GT_CXX_BUILDING_FLAGS} ${GT_CXX_OPTIMIZATION_FLAGS} ${GT_CXX_MANDATORY_FLAGS} -D${X86_BACKEND_DEFINE})
    target_link_libraries(implementation_${prec} PUBLIC common)
    add_bindings_library(implementation_wrapper_${prec} SOURCES implementation-wrapper.cpp FORTRAN_MODULE_NAME implementation_wrapper)
    target_compile_options(implementation_wrapper_${prec} PRIVATE ${GT_CXX_BUILDING_FLAGS} ${GT_CXX_OPTIMIZATION_FLAGS} ${GT_CXX_MANDATORY_FLAGS} -D${X86_BACKEND_DEFINE})
    target_link_libraries(implementation_wrapper_${prec} PUBLIC common)

    add_executable(driver driver.c)
    target_link_libraries(driver implementation_${prec}_c)
    target_compile_options(driver PRIVATE ${GT_C_BUILDING_FLAGS})

    set (CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -cpp")

    add_executable(fdriver fdriver.f90)
    target_link_libraries(fdriver implementation_${prec}_fortran)
    set_target_properties(fdriver PROPERTIES LINKER_LANGUAGE Fortran)

    add_executable(fdriver_wrapper fdriver-wrapper.f90)
    target_link_libraries(fdriver_wrapper implementation_wrapper_${prec}_fortran)
    set_target_properties(fdriver_wrapper PROPERTIES LINKER_LANGUAGE Fortran)
endif(GT_ENABLE_TARGET_X86)
