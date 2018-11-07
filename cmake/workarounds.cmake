# This function is a workaround for https://gitlab.kitware.com/cmake/cmake/issues/18558
# For CUDA, the mpi compile options are non-empty, but the flags are invalid with nvcc, so we need
# to exclude them for targets where COMPILE_LANGUAGE is not CXX
# TODO: Decide how to do that:
# - should we disable always if not CXX, or only if CUDA?
# - should we do it for the other languages too?
# - also set MPI_CXX_COMPILE_OPTIONS accordingly
function (_fix_mpi_flags)

    if (MPI_FOUND)

        get_property(mpi_compile_options TARGET MPI::MPI_CXX PROPERTY INTERFACE_COMPILE_OPTIONS)
        set (new_compile_options )
        foreach(mpi_compile_option IN LISTS mpi_compile_options)
            list (APPEND new_compile_options $<$<COMPILE_LANGUAGE:CXX>:${mpi_compile_option}>)
        endforeach()

        set_property(TARGET MPI::MPI_CXX PROPERTY INTERFACE_COMPILE_OPTIONS ${new_compile_options})

    endif()

endfunction()
