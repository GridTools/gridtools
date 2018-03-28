# -*- coding: utf-8 -*-

import textwrap

from perftest.config import default

cmake_command = default.cmake_command + [
    '-DBOOST_ROOT=/project/c14/install/kesch/boost/boost_1_66_0']


def sbatch(command):
    return textwrap.dedent(f"""\
        #!/bin/bash -l
        #SBATCH --job-name=gridtools-test
        #SBATCH --time=00:10:00
        #SBATCH --constraint=flat,quad

        srun numactl -m 1 {command}

        sync
        """)


modules = default.modules
env = default.env
make_command = default.make_command
