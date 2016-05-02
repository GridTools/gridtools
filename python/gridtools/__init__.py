import logging

from jinja2 import Environment, PackageLoader



#
# supported backends
#
BACKENDS = ('python', 'c++', 'cuda')

#
# Name of the attribute that identifies the wrapper of a user-defined stencil
# kernel
#
STENCIL_KERNEL_DECORATOR_LABEL = '__gridtools_stencil_kernel__'

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
