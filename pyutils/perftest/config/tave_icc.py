# -*- coding: utf-8 -*-

import textwrap

from perftest.config import tave_common as common
from perftest.config.tave_common import make_command, cmake_command, sbatch


modules = common.modules | {'PrgEnv-intel',
                            'craype-mic-knl'}

env = dict(common.env,
           CXX='icpc',
           CC='icc',
           CXXFLAGS='-xmic-avx512',
           OMP_NUM_THREADS=128,
           KMP_AFFINITY='balanced')

