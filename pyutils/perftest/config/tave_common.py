# -*- coding: utf-8 -*-

import textwrap

from perftest.config import default

modules = default.modules | {'/users/jenkins/easybuild/tave/modules/all/CMake/3.12.0'}

cmake_command = default.cmake_command + [
    '-DBOOST_ROOT=/project/c14/install/kesch/boost/boost_1_66_0']


def sbatch(command):
    # We allocate the KNL with 4 hyperthreads even if we only use 2 as SLURM
    # seems to select the wrong threads otherwise
    return textwrap.dedent(f"""\
        #!/bin/bash -l
        #SBATCH --job-name=gridtools-test
        #SBATCH --exclusive
        #SBATCH --nodes=1
        #SBATCH --ntasks-per-core=4
        #SBATCH --ntasks-per-node=1
        #SBATCH --cpus-per-task=256
        #SBATCH --time=00:15:00
        #SBATCH --constraint=flat,quad

        srun numactl -m 1 {command}

        sync
        """)


env = default.env
make_command = default.make_command
