# -*- coding: utf-8 -*-

from perftest.config import tave_common as default


modules = default.modules | {'PrgEnv-gnu'}

env = dict(default.env,
           CXX='g++',
           CC='gcc',
           CXXFLAGS='-march=knl -fvect-cost-model=unlimited -Ofast',
           OMP_NUM_THREADS=128,
           OMP_PROC_BIND='true',
           OMP_PLACES='{0,64}:64')

cmake_command = default.cmake_command
make_command = default.make_command
sbatch = default.sbatch
