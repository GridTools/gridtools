# -*- coding: utf-8 -*-

from perftest.config.kesch import modules, env, sbatch


modules += ['cray-libsci',
            'craype-accel-nvidia35',
            'craype-haswell',
            'craype-network-infiniband',
            'mvapich2gdr_gnu/2.2_cuda_8.0']


env.update({'CXX': 'mpicxx',
            'CC': 'mpicc'})
