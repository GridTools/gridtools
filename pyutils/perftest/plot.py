# -*- coding: utf-8 -*-

import itertools
import math
import os

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from perftest import ArgumentError, logger, result, time


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
    common, diff = result.compare([r.runtime for r in results])

    def titlestr(k, v):
        if k == 'name':
            return 'Runtime: ' + v
        if k == 'datetime':
            return 'Date/Time: ' + time.short_timestr(time.local_time(v))
        elif k == 'compiler':
            return 'Compiler: ' + os.path.basename(v).upper()
        else:
            s = str(v).title()
            if len(s) > 20:
                s = s[:20] + '…'
            return k.title() + ': ' + s

    suptitle = ', '.join(titlestr(k, v) for k, v in common.items())

    titles = ['\n'.join(titlestr(k, v) for k, v in d.items()) for d in diff]
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
    fig.suptitle(suptitle, wrap=True)

    xticks = list(range(len(results)))
    for ax, stencil, means, stdevs in itertools.zip_longest(axes, stencils,
                                                            meantimes,
                                                            stdevtimes):
        if stencil:
            ax.set_title(stencil.title())

            ax.bar(xticks, means, yerr=stdevs, width=0.8, color=colors)

            ax.set_xticks(xticks)
            ax.set_xticklabels(titles, wrap=True)
        else:
            ax.set_visible(False)

    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    return fig


def history(results, key='runtime'):
    """Plots run time history of all results."""

    stencils, meantimes, stdevtimes = result.times_by_stencil(results)

    def get_datetime(result):
        if key == 'runtime':
            datetime = result.runtime.datetime
        elif key == 'job':
            datetime = result.datetime
        else:
            raise ArgumentError('"key" argument must be "runtime" or "job"')
        return time.local_time(datetime)

    results = sorted(results, key=get_datetime)

    dates = [matplotlib.dates.date2num(get_datetime(r)) for r in results]

    if len(dates) > len(set(dates)):
        logger.warning('Non-unique datetimes in history plot')

    fig, ax = plt.subplots()

    locator = matplotlib.dates.AutoDateLocator()
    formatter = matplotlib.dates.AutoDateFormatter(locator)
    formatter.scaled[1 / 24] = '%y-%m-%d %H'
    formatter.scaled[1 / (24 * 60)] = '%y-%m-%d %H:%M'
    formatter.scaled[1 / (24 * 60 * 60)] = '%y-%m-%d %H:%M:%S'

    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    for stencil, means, stdevs in zip(stencils, meantimes, stdevtimes):
        fill_min = [m - s for m, s in zip(means, stdevs)]
        fill_max = [m + s for m, s in zip(means, stdevs)]
        ax.fill_between(dates, fill_min, fill_max, alpha=0.4)
        ax.plot(dates, means, 'o-', label=stencil.title())

    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()

    return fig
