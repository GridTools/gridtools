# -*- coding: utf-8 -*-

import os
import subprocess
import textwrap

from perftest import runtime, time


class StellaRuntime(runtime.StellaRuntimeBase):
    @property
    def version(self):
        return 'trunk'

    @property
    def datetime(self):
        posixtime = subprocess.check_output(['stat', '--format=%Y',
                                             self.path])
        return time.from_posix(posixtime)

    @property
    def compiler(self):
        if self.backend == 'cuda':
            return 'nvcc'
        else:
            return 'g++'

    @property
    def path(self):
        return os.path.join('/project', 'c14', 'install', 'kesch', 'stella',
                            'trunk_timers', f'release_{self.precision}', 'bin')


def sbatch(command):
    return textwrap.dedent(f"""\
        #!/bin/bash -l
        #SBATCH --job-name=gridtools_perftest
        #SBATCH --partition=debug
        #SBATCH --gres=gpu:1
        #SBATCH --time=00:10:00

        module load cudatoolkit/8.0.61

        export CUDA_ARCH=sm_37
        export CUDA_AUTO_BOOST=0
        export GCLOCK=875
        export G2G=1

        export OMP_PROC_BIND=true

        export MALLOC_MMAP_MAX_=0
        export MALLOC_TRIM_THRESHOLD_=536870912

        srun {command}

        sync
        """)
