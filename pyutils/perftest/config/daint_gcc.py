# -*- coding: utf-8 -*-

from perftest.config import daint_common as common
from perftest.config.daint_common import cmake_command, make_command, sbatch

modules = common.modules | {'PrgEnv-gnu'}

env = dict(common.env,
           CXX='g++',
           CC='gcc')
