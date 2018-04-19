# -*- coding: utf-8 -*-

from perftest.config import daint_common as default

modules = default.modules | {'PrgEnv-intel'}

env = dict(default.env,
           CXX='icpc',
           CC='icc')

cmake_command = default.cmake_command
make_command = default.make_command
sbatch = default.sbatch
