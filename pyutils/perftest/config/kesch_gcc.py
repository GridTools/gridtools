# -*- coding: utf-8 -*-

from perftest.config import kesch_common as common
from perftest.config.kesch_common import cmake_command, make_command, sbatch

modules = common.modules | {'cudatoolkit/8.0.61',
                            'gcc/5.4.0-2.26'}

env = dict(common.env,
           CXX='g++',
           CC='gcc')
