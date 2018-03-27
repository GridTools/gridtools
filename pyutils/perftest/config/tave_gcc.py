# -*- coding: utf-8 -*-

import textwrap

from perftest.config import tave_common as common
from perftest.config.tave_common import cmake_command, make_command, sbatch


modules = common.modules | {'PrgEnv-gnu'}

env = dict(common.env,
           CXX='g++',
           CC='gcc',
           OMP_NUM_THREADS=128,
           OMP_PROC_BIND='true',
           OMP_PLACES='{0,64}:64')
