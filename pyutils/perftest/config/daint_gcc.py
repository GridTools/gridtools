# -*- coding: utf-8 -*-

from perftest.config import daint_common as default

modules = default.modules | {'PrgEnv-gnu'}

env = dict(default.env,
           CXX='g++',
           CC='gcc')

cmake_command = default.cmake_command
make_command = default.make_command
sbatch = default.sbatch
