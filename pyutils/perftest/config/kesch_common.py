# -*- coding: utf-8 -*-

import os
import subprocess
import textwrap

from perftest import runtime, time
from perftest.config import default


modules = default.modules | {'cmake'}

env = dict(default.env,
           CUDA_ARCH='sm_37',
           CUDA_AUTO_BOOST=0,
           GCLOCK=875,
           G2G=1,
           MALLOC_MMAP_MAX_=0,
           MALLOC_TRIM_THRESHOLD_=536870912,
           OMP_PROC_BIND='true')

cmake_command = default.cmake_command
make_command = default.make_command


def sbatch(command):
    return textwrap.dedent(f"""\
        #!/bin/bash -l
        #SBATCH --job-name=gridtools-test
        #SBATCH --partition=debug
        #SBATCH --gres=gpu:1
        #SBATCH --time=00:10:00

        srun {command}

        sync
        """)


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
