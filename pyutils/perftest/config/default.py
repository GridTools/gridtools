# -*- coding: utf-8 -*-

import textwrap


modules = set()

env = dict()

cmake_command = ['cmake',
                 '-DGT_ENABLE_PYUTILS=ON',
                 '-DGT_ENABLE_PERFORMANCE_METERS=ON']

make_command = ['make', '-j8']


def sbatch(command):
    return textwrap.dedent(f"""\
        #!/bin/bash -l
        #SBATCH --job-name=gridtools-test

        srun {command}
        """)
