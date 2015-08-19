import logging

from jinja2 import Environment, PackageLoader



#
# supported backends
#
BACKENDS = ('python', 'c++', 'cuda')

#
# Specifying here a list of tests to perform (this will be read by tests.tests_envvar when executed from python_run_tests.sh)
#
TEST_LIST = ('tests.test_stencils')


#
# Enviroment variables associated to backends 
# BASIC_ENVVARS is defined by using a tuple (cannot be modified in the code)
#
BASIC_ENVVARS = ('GRIDTOOLS_ROOT', 'BOOST_ROOT', 'CXX')
#
# In case backend is cuda one more environment variable is needed.
# A tuple is defined by starting from BASIC_ENVVARS and adding the env var related to cuda
# 
CUDA_ENVVARS_tmp = list(BASIC_ENVVARS)
CUDA_ENVVARS_tmp.append('CUDATOOLKIT_HOME')
CUDA_ENVVARS = tuple(CUDA_ENVVARS_tmp)


#
# initialize the template renderer environment
#
logging.debug ("Initializing the template environment ...")

def join_with_prefix (a_list, prefix, attribute=None):
    """
    A custom filter for Jinja template rendering.-
    """
    if attribute is None:
        return ['%s%s' % (prefix, e) for e in a_list]
    else:
        return ['%s%s' % (prefix, getattr (e, attribute)) for e in a_list]

JinjaEnv = Environment (loader=PackageLoader ('gridtools',
                                              'templates'))
JinjaEnv.filters["join_with_prefix"] = join_with_prefix


#
# plotting environment
#
logging.info ("Initializing plotting environment 'gridtools.plt' ...")
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import axes3d
except ImportError:
    plt = None
    logging.error ("Matplotlib not found: plotting is not available")
