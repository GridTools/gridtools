#-*- coding: utf-8 -*-
import os
import sys
from setuptools import setup


version = "0.1.1"


#
# this interface is Python 3.x only
#
if sys.version_info.major < 3:
    print ("Python 3.x is required.")
    sys.exit (1)


def read_file (fname):
    """
    Reads file into string.
    :param fname: full path to the file
    :return:      file content as a string.-
    """
    return open (os.path.join (os.path.dirname (__file__), fname)).read ( )


setup (
    name             = 'gridtools4py',
    description      = 'Python interface C++ library Gridtools',
    long_description = read_file ('README.md'),
    version          = version,
    classifiers      = ['Development Status :: 5 - Production/Stable',
                        'Environment :: Console',
                        'Environment :: Web Environment',
                        'Environment :: X11 Applications',
                        'Environment :: MacOS X',
                        'Operating System :: OS Independent',
                        'Programming Language :: Python',
                        'Topic :: Software Development :: Libraries :: Python Modules',
                        'Topic :: Utilities',
                        'Programming Language :: Python :: 3',
                        'Programming Language :: Python :: 3.3',
                        'Programming Language :: Python :: 3.4',],
    keywords         = 'stencil jit c++ openmp cuda',
    author           = 'Lucas Benedicic',
    author_email     = 'benedicic@cscs.ch',
    url              = 'https://github.com/eth-cscs/gridtools',
    license          = 'COSMO',
    packages         = ['gridtools', 'tests'],
    package_data     = {         '' : ['README.md'],
                        'gridtools' : ['templates/*'],
                            'tests' : ['*.npy']},
    install_requires = read_file ('requirements.txt').split ('\n')
)
