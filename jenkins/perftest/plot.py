# -*- coding: utf-8 -*-

import itertools
import math

import cycler
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


prop_cycle = cycler.cycler('color', ['#1f77b4', '#ff7f0e',
                                     '#2ca02c', '#d62728',
                                     '#9467bd', '#8c564b',
                                     '#e377c2', '#7f7f7f',
                                     '#bcbd22', '#17becf'])


def figsize(rows=1, cols=1):
    return (10 * rows, 5 * cols)


def discrete_colors(n):
    return [c['color'] for c in itertools.islice(iter(prop_cycle), n)]


def get_titles(*results):
    common_fields = results[0].common_fields(*results[1:])
    common_fields.discard('times')

    def title(result, fields, sep):
        return sep.join(str(getattr(result, f)).title() for f in fields)

    suptitle = title(results[0], common_fields, ', ')

    common_fields.add('times')
    titles = [title(r, r.fields - common_fields, '\n') for r in results]

    return suptitle, titles


def compare(*results):
    assert results[0].has_same_stencils(*results[1:])
    ntimes = len(results[0].times)

    rows = math.floor(math.sqrt(ntimes))
    cols = math.ceil(ntimes / rows)

    fig, axarr = plt.subplots(rows, cols, squeeze=False,
                              figsize=figsize(rows, cols))

    stencils = results[0].times.keys()
    stimes = {s: [r.times[s] for r in results] for s in stencils}

    axes = itertools.chain(*axarr)
    colors = discrete_colors(len(results))

    suptitle, titles = get_titles(*results)
    fig.suptitle(suptitle)

    xticks = list(range(ntimes))
    for ax, (stencil, times) in zip(axes, stimes.items()):
        ax.set_title(stencil.title())

        ax.bar(xticks, times, width=0.8, color=colors)

        ax.set_xticks(xticks)
        ax.set_xticklabels(titles)

    fig.tight_layout()
    return fig
