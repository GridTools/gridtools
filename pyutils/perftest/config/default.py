# -*- coding: utf-8 -*-

import textwrap


modules = set()

env = dict()

cmake_command = ['cmake',
                 '-DENABLE_PYUTILS=ON',
                 '-DENABLE_PERFORMANCE_METERS=ON']

make_command = ['make', '-j8']


def sbatch(command):
    return textwrap.dedent(f"""\
        #!/bin/bash -l
        #SBATCH --job-name=gridtools-test

        srun {command}
        """)
