# -*- coding: utf-8 -*-

import itertools
import math

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from perftest import ArgumentError, logger, result, utils


plt.style.use('ggplot')

prop_cycle = matplotlib.rcParams['axes.prop_cycle']


def figsize(rows=1, cols=1):
    """Default figsize for a plot with given subplot rows and columns."""
    return (7 * rows, 5 * cols)


def discrete_colors(n):
    """Generates `n` discrete colors for plotting."""
    return [c['color'] for c in itertools.islice(iter(prop_cycle), n)]


def get_titles(results):
    """Generates plot titles."""
    common, diff = result.compare(r.runtime for r in results)

    def titlestr(v):
        try:
            return utils.short_timestr(v)
        except ArgumentError:
            s = str(v).title()
            return s if len(s) < 30 else s[:27] + '...'

    suptitle = ', '.join(titlestr(v) for v in common.values())

    titles = ['\n'.join(titlestr(v) for v in d.values()) for d in diff]
    return suptitle, titles


def compare(results):
    """Plots run time comparison of all results, one subplot per stencil."""

    stencils, meantimes, stdevtimes = result.times_by_stencil(results)

    rows = math.floor(math.sqrt(len(stencils)))
    cols = math.ceil(len(stencils) / rows)

    fig, axarr = plt.subplots(rows, cols, squeeze=False,
                              figsize=figsize(cols * len(results) / 2, rows))

    axes = itertools.chain(*axarr)
    colors = discrete_colors(len(results))

    suptitle, titles = get_titles(results)
    fig.suptitle(suptitle)

    xticks = list(range(len(results)))
    for ax, stencil, means, stdevs in itertools.zip_longest(axes, stencils,
                                                            meantimes,
                                                            stdevtimes):
        if stencil:
            ax.set_title(stencil.title())

            ax.bar(xticks, means, yerr=stdevs, width=0.8, color=colors)

            ax.set_xticks(xticks)
            ax.set_xticklabels(titles)
        else:
            ax.set_visible(False)

    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    return fig


def history(results, key='runtime'):
    """Plots run time history of all results."""

    stencils, meantimes, stdevtimes = result.times_by_stencil(results)

    def get_date(result):
        if key == 'runtime':
            datetime = result.runtime.datetime
        elif key == 'job':
            datetime = result.datetime
        else:
            raise ArgumentError('"key" argument must be "runtime" or "job"')
        return matplotlib.dates.date2num(datetime)

    dates = [get_date(r) for r in results]

    if len(dates) > len(set(dates)):
        logger.warning('Non-unique datetimes in history plot')

    fig, ax = plt.subplots()

    locator = matplotlib.dates.AutoDateLocator()
    ax.xaxis.set_major_formatter(matplotlib.dates.AutoDateFormatter(locator))
    ax.xaxis.set_major_locator(locator)

    for stencil, means, stdevs in zip(stencils, meantimes, stdevtimes):
        ax.errorbar(dates, means, yerr=stdevs, fmt='o-', label='stencil')

    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()

    return fig
