macro(detect_fortran_compiler)
    # Enable Fortran here, as enabling locally might be undefined,
    # see https://cmake.org/cmake/help/v3.12/command/enable_language.html
    # "... it must be called in the highest directory common to all targets using the named language..."
    include(CheckLanguage)
    check_language(Fortran)
    if (CMAKE_Fortran_COMPILER)
        enable_language(Fortran)
    else()
        message(WARNING "Fortran Compiler has not been found. Tests using fortran will not be built!")
    endif()
endmacro(detect_fortran_compiler)

macro(detect_c_compiler)
    # Enable C here, as enabling locally might be undefined,
    # see https://cmake.org/cmake/help/v3.12/command/enable_language.html
    # "... it must be called in the highest directory common to all targets using the named language..."
    include(CheckLanguage)
    check_language(C)
    if (CMAKE_C_COMPILER)
        enable_language(C)
    else()
        message(WARNING "C Compiler has not been found. Tests using C will not be built!")
    endif()
endmacro(detect_c_compiler)
