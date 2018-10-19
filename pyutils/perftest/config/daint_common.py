# -*- coding: utf-8 -*-

import textwrap
import os

from perftest.config import default

modules = default.modules | {'daint-gpu',
                             'cudatoolkit/9.1.85_3.18-6.0.7.0_5.1__g2eb7c52',
                             'CMake'}

env = dict(default.env,
           CUDA_ARCH='sm_60',
           CUDA_AUTO_BOOST=0,
           OMP_PROC_BIND='true',
           OMP_NUM_THREADS=24)

cmake_command = default.cmake_command + [
    '-DBOOST_ROOT=' + os.environ['SCRATCH'] + '/../jenkins/install/boost/boost_1_67_0']
make_command = ['srun', '--constraint=gpu', '--account=c14',
                '--time=00:30:00', 'make', '-j24']


def sbatch(command):
    return textwrap.dedent(f"""\
        #!/bin/bash -l
        #SBATCH --job-name=gridtools-test
        #SBATCH --partition=cscsci
        #SBATCH --nodes=1
        #SBATCH --ntasks-per-core=2
        #SBATCH --ntasks-per-node=1
        #SBATCH --cpus-per-task=24
        #SBATCH --constraint=gpu
        #SBATCH --time=00:15:00
        #SBATCH --account=c14

        srun {command}

        sync
        """)
