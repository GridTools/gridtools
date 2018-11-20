# This is a workaround to properly detect SLURM on our machines, respectively,
# we want to give srun priority over mpirun / mpiexec. If we use slurm, we
# assume that at runtime, we execute in an allocated environment
function (set_duplicated_var new old value)
    get_property(_advanced CACHE ${old} PROPERTY ADVANCED)
    get_property(_helpstring CACHE ${old} PROPERTY HELPSTRING)
    get_property(_strings CACHE ${old} PROPERTY STRINGS)
    get_property(_type CACHE ${old} PROPERTY TYPE)
    get_property(_value CACHE ${old} PROPERTY VALUE)

    set(${new} ${value} CACHE ${_type} "${_helpstring}")
    set_property(CACHE ${new} PROPERTY ADVANCED "${_advanced}")
    set_property(CACHE ${new} PROPERTY STRINGS "${_strings}")
endfunction()

function(_fix_mpi_exec)

    # Note: This is the same command that is used by CTest
    find_program(SLURM_SRUN_COMMAND
                srun
                DOC
                "Path to the SLURM srun executable")

    if (SLURM_SRUN_COMMAND)
        set_duplicated_var(MPITEST_EXECUTABLE MPIEXEC_EXECUTABLE "${SLURM_SRUN_COMMAND}")
        set_duplicated_var(MPITEST_MAX_NUMPROCS MPIEXEC_MAX_NUMPROCS "")
        set_duplicated_var(MPITEST_NUMPROC_FLAG MPIEXEC_NUMPROC_FLAG "-n")
        set_duplicated_var(MPITEST_POSTFLAGS MPIEXEC_POSTFLAGS "")
        set_duplicated_var(MPITEST_PREFLAGS MPIEXEC_PREFLAGS "")
        set(use_mpi_wrappers ON)
    else ()
        set_duplicated_var(MPITEST_EXECUTABLE MPIEXEC_EXECUTABLE "${MPIEXEC_EXECUTABLE}")
        set_duplicated_var(MPITEST_MAX_NUMPROCS MPIEXEC_MAX_NUMPROCS "${MPIEXEC_MAX_NUMPROCS}")
        set_duplicated_var(MPITEST_NUMPROC_FLAG MPIEXEC_NUMPROC_FLAG "${MPIEXEC_NUMPROC_FLAG}")
        set_duplicated_var(MPITEST_POSTFLAGS MPIEXEC_POSTFLAGS "${MPIEXEC_POSTFLAGS}")
        set_duplicated_var(MPITEST_PREFLAGS MPIEXEC_PREFLAGS "${MPIEXEC_PREFLAGS}")
        set(use_mpi_wrappers OFF)
    endif()

    set( MPITEST_HOST_ENVIRONMENT "$ENV{MPI_HOST_JOB_ENV}" CACHE STRING "Environment used for MPI Host tests in CTest" )
    set( MPITEST_CUDA_ENVIRONMENT "$ENV{MPI_CUDA_JOB_ENV}" CACHE STRING "Environment used for MPI CUDA tests in CTest" )
    set( TEST_HOST_ENVIRONMENT "$ENV{HOST_JOB_ENV}" CACHE STRING "Environment used for non-MPI Host tests in CTest" )
    set( TEST_CUDA_ENVIRONMENT "$ENV{CUDA_JOB_ENV}" CACHE STRING "Environment used for non-MPI CUDA tests in CTest" )
    option( TEST_USE_WRAPPERS_FOR_ALL_TESTS "Use mpi wrappers for all tests" ${use_mpi_wrappers})
    mark_as_advanced(
        MPITEST_HOST_ENVIRONMENT
        MPITEST_CUDA_ENVIRONMENT
        TEST_HOST_ENVIRONMENT
        TEST_CUDA_ENVIRONMENT
        TEST_USE_WRAPPERS_FOR_ALL_TESTS)

endfunction()
