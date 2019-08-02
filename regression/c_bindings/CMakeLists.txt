if (GT_ENABLE_BACKEND_X86)
    if(GT_SINGLE_PRECISION)
        set(prec float)
    else()
        set(prec double)
    endif()

    bindgen_add_library(implementation_${prec} SOURCES implementation.cpp FORTRAN_MODULE_NAME implementation)
    target_link_libraries(implementation_${prec} PUBLIC GridToolsTestX86)
    bindgen_enable_fortran_library(implementation_${prec})

    bindgen_add_library(implementation_wrapper_${prec} SOURCES implementation_wrapper.cpp FORTRAN_MODULE_NAME implementation_wrapper)
    target_link_libraries(implementation_wrapper_${prec} PUBLIC GridToolsTestX86)
    bindgen_enable_fortran_library(implementation_wrapper_${prec})

    if (CMAKE_C_COMPILER_LOADED)
        add_executable(driver driver.c)
        target_link_libraries(driver implementation_${prec}_c)
    endif()

    if (CMAKE_Fortran_COMPILER_LOADED)
        include(fortran_helpers)

        add_executable(fdriver fdriver.f90)
        target_link_libraries(fdriver implementation_${prec}_fortran)
        set_target_properties(fdriver PROPERTIES LINKER_LANGUAGE Fortran)
        bindgen_enable_fortran_preprocessing_on_target(fdriver)

        add_executable(fdriver_wrapper fdriver_wrapper.f90)
        target_link_libraries(fdriver_wrapper implementation_wrapper_${prec}_fortran)
        set_target_properties(fdriver_wrapper PROPERTIES LINKER_LANGUAGE Fortran)
        bindgen_enable_fortran_preprocessing_on_target(fdriver_wrapper)
    endif()
endif(GT_ENABLE_BACKEND_X86)
