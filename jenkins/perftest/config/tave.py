# -*- coding: utf-8 -*-

import os
import subprocess
import textwrap

from perftest import runtime, time


class GridtoolsRuntime(runtime.GridtoolsRuntimeBase):
    @property
    def path(self):
        return os.path.join('/scratch', 'snx2000', 'jenkins',
                            'workspace', f'GridTools_{self.grid}_PR',
                            'build_type', 'release',
                            'compiler', 'icc',
                            'label', 'tave',
                            'mpi', 'MPI',
                            'real_type', self.precision,
                            'target', 'cpu',
                            'build')


def sbatch(command):
    return textwrap.dedent(f"""\
        #!/bin/bash -l
        #SBATCH --job-name=gridtools_perftest
        #SBATCH --time=00:15:00
        #SBATCH --constraint=flat,quad
        #SBATCH --reservation=knl

        module switch PrgEnv-cray PrgEnv-intel
        module load craype-mic-knl

        export KMP_AFFINITY=balanced

        srun numactl -m 1 {command}

        sync
        """)
