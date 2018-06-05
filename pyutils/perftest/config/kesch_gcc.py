# -*- coding: utf-8 -*-

from perftest.config import kesch_common as default

modules = default.modules | {'cudatoolkit/8.0.61',
                             'gcc/5.4.0-2.26'}

env = dict(default.env,
           CXX='g++',
           CC='gcc')

cmake_command = default.cmake_command
make_command = default.make_command
sbatch = default.sbatch
StellaRuntime = default.StellaRuntime
