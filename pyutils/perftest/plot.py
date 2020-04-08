# -*- coding: utf-8 -*-

import collections
import contextlib
import functools
import itertools
import math
import operator
import os
import pathlib
import re
import sys
import types
import warnings
from xml.etree import ElementTree as et

import dateutil.parser
import matplotlib
from matplotlib import pyplot as plt
import numpy as np

from pyutils import log
from perftest import html

plt.style.use('ggplot')


def _compare_medians(a, b, n=1000, alpha=0.05):
    scale = np.median(a)
    a = np.asarray(a) / scale
    b = np.asarray(b) / scale
    # bootstrap sampling
    asamps = np.random.choice(a, (a.size, n))
    bsamps = np.random.choice(b, (b.size, n))
    # bootstrap estimates of difference of medians
    bootstrap_estimates = np.median(bsamps, axis=0) - np.median(asamps, axis=0)
    # percentile bootstrap confidence interval
    ci = np.quantile(bootstrap_estimates, [alpha / 2, 1 - alpha / 2])
    log.debug(f'Boostrap results (n = {n}, alpha = {alpha})',
              f'{ci[0]:8.5f} - {ci[1]:8.5f}')
    return ci


def _classify(ci):
    lower, upper = ci
    assert lower <= upper

    # large uncertainty
    if upper - lower > 0.1:
        return '??'

    # no change
    if -0.01 <= lower <= 0 <= upper <= 0.01:
        return '='
    if -0.02 <= lower <= upper <= 0.02:
        return '(=)'

    # probably no change, but quite large uncertainty
    if -0.05 <= lower <= 0 <= upper <= 0.05:
        return '?'

    # faster
    if -0.01 <= upper <= 0.0:
        return '(+)'
    if -0.05 <= upper <= -0.01:
        return '+'
    if -0.1 <= upper <= -0.05:
        return '++'
    if upper <= -0.1:
        return '+++'

    # slower
    if 0.01 >= lower >= 0.0:
        return '(-)'
    if 0.05 >= lower >= 0.01:
        return '-'
    if 0.1 >= lower >= 0.05:
        return '--'
    if lower >= 0.1:
        return '---'

    # no idea
    return '???'


def _significant(ci):
    return '=' not in _classify(ci)


def _css_class(class_str):
    if '-' in class_str:
        return 'bad'
    if '?' in class_str:
        return 'unknown'
    if '+' in class_str:
        return 'good'
    return ''


def _add_comparison_table(report, results):
    log.debug('Generating comparison table')

    def name_backend(result):
        return result['name'], result['backend']

    backends = set()
    names = set()
    classified = collections.defaultdict(dict)
    for (name, backend), items in itertools.groupby(sorted(results,
                                                           key=name_backend),
                                                    key=name_backend):
        data = {item['float_type']: _classify(item['ci']) for item in items}
        classified[name][backend] = data['float'], data['double']
        names.add(name)
        backends.add(backend)

    backends = list(sorted(backends))
    names = list(sorted(names))

    with report.table('Comparison') as table:
        with table.row() as row:
            row.cell('BENCHMARK')
            for backend in backends:
                row.cell(backend.upper())

        for name in names:
            with table.row() as row:
                row.cell(name.replace('_', ' ').title())
                for backend in backends:
                    try:
                        clss, double_clss = classified[name][backend]
                        if double_clss != clss:
                            clss += ' ' + double_clss
                    except KeyError:
                        clss = ''
                    row.cell(clss).set('class', _css_class(clss))
    log.debug('Generated performance comparison table')


def _histogram_plot(title, result, output):
    fig, ax = plt.subplots(figsize=(10, 5))
    before = result['series_before']
    after = result['series_after']
    bins = np.linspace(0, max(np.amax(before), np.amax(after)), 50)
    ax.hist(before, alpha=0.5, bins=bins, density=True, label='Before')
    ax.hist(after, alpha=0.5, bins=bins, density=True, label='After')
    style = iter(plt.rcParams['axes.prop_cycle'])
    ax.axvline(np.median(before), **next(style))
    ax.axvline(np.median(after), **next(style))
    ax.legend(loc='upper left')
    ax.set_xlabel('Time [s]')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output)
    log.debug(f'Sucessfully written histogram plot to {output}')
    plt.close(fig)


def _add_comparison_info(report, before, after):
    with report.table('Info') as table:
        with table.row() as row:
            row.fill('Property', 'Before', 'After')

        for k, vafter in after['gridtools'].items():
            vbefore = before['gridtools'].get(k, '')
            with table.row() as row:
                row.fill('GridTools ' + k.title(), vbefore, vafter)

        for k, vafter in after['environment'].items():
            vbefore = before['environment'].get(k, '')
            with table.row() as row:
                row.fill(k.title(), vbefore, vafter)


def _compare(a, b):
    def same_keys(ao, bo):
        return all(ao[k] == v for k, v in bo.items() if k != 'series')

    results = []
    for props in b['outputs']:
        b_series = props.pop('series')
        try:
            a_series = next(o for o in a['outputs'] if all(
                o[k] == v for k, v in props.items()))['series']
        except StopIteration:
            log.debug('Nothing to compare for', props)
            continue

        props['ci'] = _compare_medians(a_series, b_series)
        props['series_before'] = a_series
        props['series_after'] = b_series
        results.append(props)

    return results


def _add_comparison_plots(report, results):
    with report.image_grid('Details') as grid:
        for result in results:
            if _significant(result['ci']):
                title = (result['name'].replace('_', ' ').title() + ' (' +
                         result['backend'].upper() + ', ' +
                         result['float_type'].upper() + ', ' +
                         _classify(result['ci']) + ')')
                _histogram_plot(title, result, grid.image())


def compare(before, after, output):
    with html.Report(output, 'GridTools Performance Test Results') as report:
        results = _compare(before, after)

        _add_comparison_table(report, results)
        _add_comparison_plots(report, results)
        _add_comparison_info(report, before, after)


_OUTPUT_KEYS = 'name', 'backend', 'float_type'


def _output_key(output):
    return tuple(output[k] for k in _OUTPUT_KEYS)


def _outputs_by_key(data):
    return {_output_key(o): o['series'] for o in data['outputs']}


def _history_data(data, key, limit):
    def get_datetime(result):
        source = 'gridtools' if key == 'commit' else 'environment'
        return dateutil.parser.isoparse(result[source]['datetime'])

    data = sorted(data, key=get_datetime)
    if limit:
        data = data[-limit:]

    datetimes = [get_datetime(d) for d in data]
    outputs = [_outputs_by_key(d) for d in data]

    keys = set.union(*(set(o.keys()) for o in outputs))

    measurement = collections.namedtuple('measurment',
                                         ['median', 'lower', 'upper'])
    measurements = {k: measurement([], [], []) for k in keys}
    for o in outputs:
        for k in keys:
            try:
                lower, median, upper = np.percentile(o[k], [5, 50, 95])
            except KeyError:
                lower = median = upper = np.nan
            measurements[k].lower.append(lower)
            measurements[k].median.append(median)
            measurements[k].upper.append(upper)

    return datetimes, measurements


def _history_plot(title, dates, measurements, output):
    fig, ax = plt.subplots(figsize=(10, 5))
    dates = [matplotlib.dates.date2num(d) for d in dates]
    if len(dates) > len(set(dates)):
        log.warning('Non-unique dates in history plot')

    locator = matplotlib.dates.AutoDateLocator()
    formatter = matplotlib.dates.AutoDateFormatter(locator)
    formatter.scaled[1 / 24] = '%y-%m-%d %H:%M'
    formatter.scaled[1 / (24 * 60)] = '%y-%m-%d %H:%M'
    formatter.scaled[1 / (24 * 60 * 60)] = '%y-%m-%d %H:%M:%S'

    ax.set_title(title)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    ax.fill_between(dates, measurements.lower, measurements.upper, alpha=0.2)
    ax.plot(dates, measurements.median, '|-')
    ax.set_ylim(bottom=0)
    ax.set_ylabel('Time [s]')
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    log.debug(f'Sucessfully written history plot to {output}')
    plt.close(fig)


def history(data, output, key='job', limit=None):
    with html.Report(output, 'GridTools Performance History') as report:
        dates, measurements = _history_data(data, key, limit)

        with report.image_grid() as grid:
            for k, m in measurements.items():
                title = ', '.join(k).replace('_', ' ').title()
                _history_plot(title, dates, m, grid.image())
