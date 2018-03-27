# -*- coding: utf-8 -*-

import textwrap

from perftest.config import tave_common as common
from perftest.config.tave_common import cmake_command, make_command, sbatch


modules = common.modules | {'PrgEnv-intel',
                            'craype-mic-knl'}

env = dict(common.env,
           CXX='icpc',
           CC='icc',
           KMP_AFFINITY='balanced')
