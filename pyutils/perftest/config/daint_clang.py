# -*- coding: utf-8 -*-

from perftest.config import daint_common as default

modules = default.modules | {'/users/vogtha/modules/compilers/clang/3.8.1'}

env = dict(default.env,
           CXX='clang++',
           CC='clang')

cmake_command = default.cmake_command
make_command = default.make_command
sbatch = default.sbatch
