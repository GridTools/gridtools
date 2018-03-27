# -*- coding: utf-8 -*-

import textwrap

from perftest.config.default import modules, env, cmake_command, make_command


def sbatch(command):
    return textwrap.dedent(f"""\
        #!/bin/bash -l
        #SBATCH --job-name=gridtools-test
        #SBATCH --time=00:10:00
        #SBATCH --constraint=flat,quad

        srun numactl -m 1 {command}

        sync
        """)
