# -*- coding: utf-8 -*-

import textwrap

from perftest.config import default


modules = default.modules | {'daint-gpu',
                             'cudatoolkit',
                             'CMake'}

env = dict(default.env,
           CUDA_ARCH='sm_60',
           CUDA_AUTO_BOOST=0,
           OMP_PROC_BIND='true')

cmake_command = default.cmake_command + [
    '-DBOOST_ROOT=/scratch/snx3000/jenkins/install/boost/boost_1_66_0']
make_command = ['srun', '--constraint=gpu', '--account=c14',
                '--time=00:15:00', 'make', '-j24']


def sbatch(command):
    return textwrap.dedent(f"""\
        #!/bin/bash -l
        #SBATCH --job-name=gridtools-test
        #SBATCH --partition=normal
        #SBATCH --gres=gpu:1
        #SBATCH --time=00:10:00

        srun {command}

        sync
        """)
