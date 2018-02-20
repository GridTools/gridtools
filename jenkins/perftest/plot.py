# -*- coding: utf-8 -*-

from datetime import datetime
import itertools
import math

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from perftest import result, logger


plt.style.use('ggplot')

prop_cycle = matplotlib.rcParams['axes.prop_cycle']


def figsize(rows=1, cols=1):
    return (10 * rows, 5 * cols)


def discrete_colors(n):
    return [c['color'] for c in itertools.islice(iter(prop_cycle), n)]


def get_titles(*results):
    common, diff = results[0].runtime.compare(*(r.runtime for r in
                                                results[1:]))

    def titlestr(v):
        if isinstance(v, datetime):
            return v.strftime('%y-%m-%d %H:%M')
        else:
            s = str(v).title()
            return s if len(s) < 30 else s[:27] + '...'

    suptitle = ', '.join(titlestr(v) for v in common.values())

    titles = ['\n'.join(titlestr(v) for v in vs) for vs in zip(*diff.values())]
    return suptitle, titles


def compare(*results):
    stencils, meantimes, stdevtimes = result.times_by_stencil(*results)

    rows = math.floor(math.sqrt(len(stencils)))
    cols = math.ceil(len(stencils) / rows)

    fig, axarr = plt.subplots(rows, cols, squeeze=False,
                              figsize=figsize(rows, cols))

    axes = itertools.chain(*axarr)
    colors = discrete_colors(len(results))

    suptitle, titles = get_titles(*results)
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


def history(*results):
    stencils, meantimes, stdevtimes = result.times_by_stencil(*results)
    dates = [matplotlib.dates.date2num(r.runtime.datetime) for r in results]

    if len(dates) > len(set(dates)):
        logger.warning('Non-unique datetimes in history plot')

    fig, ax = plt.subplots()

    for stencil, means, stdevs in zip(stencils, meantimes, stdevtimes):
        ax.plot_date(dates, means, 'o-', label=stencil)
        ax.errorbar(dates, means, yerr=stdevs)

    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()

    return fig
