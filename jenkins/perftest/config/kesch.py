# -*- coding: utf-8 -*-

import os
import subprocess
import textwrap

from perftest import utils, runtime


class StellaRuntime(runtime.StellaRuntimeBase):
    @property
    def version(self):
        return 'trunk'

    @property
    def datetime(self):
        posixtime = subprocess.check_output(['stat', '--format=%Y',
                                             self.path])
        return utils.timestr_from_posix(posixtime)

    @property
    def path(self):
        return os.path.join('/project', 'c14', 'install', 'kesch', 'stella',
                            'trunk_timers', f'release_{self.precision}', 'bin')


class GridtoolsRuntime(runtime.GridtoolsRuntimeBase):
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
        #SBATCH --job-name=gridtools_perftest
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
        export CUDA_AUTO_BOOST=0
        export GCLOCK=875
        export G2G=1

        export OMP_PROC_BIND=true
        export OMP_PLACES=threads
        export OMP_NUM_THREADS=12

        export MALLOC_MMAP_MAX_=0
        export MALLOC_TRIM_THRESHOLD_=536870912

        echo "Running on node $HOSTNAME"
        echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

        srun {command}

        sync
        """)
