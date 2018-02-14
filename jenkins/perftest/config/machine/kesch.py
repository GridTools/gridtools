# -*- coding: utf-8 -*-

import os
import textwrap


def stella_path(grid_type, precision, backend):
    return os.path.join('/project', 'c14', 'install', 'kesch', 'stella',
                        'trunk_timers', f'release_{precision}', 'bin')


def gridtools_path(grid_type, precision, backend):
    return os.path.join('/scratch', 'jenkins', 'workspace',
                        f'GridTools_{grid_type}_PR', 'build_type', 'release',
                        'compiler', 'gcc', 'label', 'kesch', 'mpi', 'MPI',
                        'real_type', precision, 'target', 'gpu', 'build')


def sbatch(command):
    return textwrap.dedent(f"""\
        #!/bin/bash -l
        #SBATCH --job-name=gridtools_test
        #SBATCH --nodes=1
        #SBATCH --ntasks=1
        #SBATCH --ntasks-per-node=1
        #SBATCH --partition=debug
        #SBATCH --time=00:15:00
        #SBATCH --gres=gpu:1
        #SBATCH --cpus-per-task=12

        module load craype-network-infiniband
        module load craype-haswell
        module load craype-accel-nvidia35
        module load cray-libsci
        module load cudatoolkit/8.0.61
        module load mvapich2gdr_gnu/2.2_cuda_8.0
        module load gcc/5.4.0-2.26
        module load cmake/3.9.1

        export Boost_NO_SYSTEM_PATHS=true
        export Boost_NO_BOOST_CMAKE=true
        export BOOST_ROOT=/project/c14/install/kesch/boost/boost_1_66_0
        export BOOST_INCLUDE=/project/c14/install/kesch/boost/boost_1_66_0/include/
        export CUDA_ARCH=sm_37

        {command}

        sync
        """)
