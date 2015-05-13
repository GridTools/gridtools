import logging
from jinja2 import Environment, PackageLoader


def join_with_prefix (a_list, prefix, attribute=None):
    """
    A custom filter for template rendering.-
    """
    if attribute is None:
        return ['%s%s' % (prefix, e) for e in a_list]
    else:
        return ['%s%s' % (prefix, getattr (e, attribute)) for e in a_list]

#
# initialize the template renderer environment
#
JinjaEnv = Environment (loader=PackageLoader ('gridtools',
                                              'templates'))
JinjaEnv.filters["join_with_prefix"] = join_with_prefix


#
# plotting environment
#
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import axes3d

except ImportError:
    logging.error ("Matplotlib not found: plotting is not available")

else:
    fig = plt.figure ( )
    ax  = fig.add_subplot (111,
                           projection='3d',
                           autoscale_on=False)
