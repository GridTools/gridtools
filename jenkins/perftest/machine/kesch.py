# -*- coding: utf-8 -*-

import os
import textwrap

from perftest.runtime import StellaRuntimeBase, GridtoolsRuntimeBase


class StellaRuntime(StellaRuntimeBase):
    @property
    def version(self):
        return 'trunk'

    @property
    def path(self):
        return os.path.join('/project', 'c14', 'install', 'kesch', 'stella',
                            'trunk_timers', f'release_{self.precision}', 'bin')


class GridtoolsRuntime(GridtoolsRuntimeBase):
    @property
    def path(self):
        return os.path.join('/scratch', 'jenkins',
                            'workspace', f'GridTools_{self.grid}_PR',
                            'build_type', 'release',
                            'compiler', 'gcc',
                            'label', 'kesch',
                            'mpi', 'MPI',
                            'real_type', self.precision,
                            'target', 'gpu',
                            'build')


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

        export CUDA_ARCH=sm_37

        {command}

        sync
        """)
