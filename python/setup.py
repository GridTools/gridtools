#-*- coding: utf-8 -*-
import sys
from setuptools import setup



version = "0.0.1"


#
# this interface is Python 3.x only
#
if sys.version_info.major < 3:
    print ("Python 3.x is required for the interface.")
    sys.exit (1)



def read (fname):
    """
    Utility function to read the README file; used for the long_description.-
    """
    return open (os.path.join (os.path.dirname (__file__), fname)).read ( )



if __name__ == '__main__':
    setup (
        name='gridtools4py',
        description="Python interface C++ library Gridtools",
        version=version,
        author="Lucas Benedicic",
        author_email="benedicic@cscs.ch",
        keywords="stencil jit c++ openmp cuda",
        package_dir={'': '.'},
        packages=["gridtools"],
        #install_requires=['Jinja2', 'matplotlib', 'six', 'pytz', 'numpy'],
        install_requires=['Jinja2', 'numpy'],
        test_suite="tests",
        license="COSMO",
        url="https://github.com/eth-cscs/gridtools",
    )
