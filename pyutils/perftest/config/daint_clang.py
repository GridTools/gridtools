# -*- coding: utf-8 -*-

from perftest.config import daint_common as common
from perftest.config.daint_common import cmake_command, make_command, sbatch

modules = common.modules | {'/users/vogtha/modules/compilers/clang/3.8.1'}

env = dict(common.env,
           CXX='clang++',
           CC='clang')
