# -*- coding: utf-8 -*-

from perftest.config import tave_common as default


modules = default.modules | {'PrgEnv-intel',
                             'craype-mic-knl'}

env = dict(default.env,
           CXX='icpc',
           CC='icc',
           CXXFLAGS='-xmic-avx512',
           OMP_NUM_THREADS=128,
           KMP_AFFINITY='balanced')

cmake_command = default.cmake_command
make_command = default.make_command
sbatch = default.sbatch
