# -*- coding: utf-8 -*-

from collections import defaultdict
import functools
import itertools
import math
import operator
import os
import re
import sys
import types
import warnings
from xml.etree import ElementTree as et

import numpy as np

from pyutils import log

from plotly import graph_objs as go, colors


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


def _color(class_str):
    red = np.array([180.0, 30.0, 0.0])
    green = np.array([100.0, 180.0, 0.0])
    blue = np.array([0.0, 100.0, 180.0])
    white = np.array([245.0] * 3)

    p = class_str.count('+') / 3
    m = class_str.count('-') / 3
    u = class_str.count('?') / 3
    if '(' in class_str:
        p *= 0.5
        m *= 0.5
        u *= 0.5
    w = 1 - p - m - u
    assert w >= 0
    return red * m + green * p + blue * u + white * w


def _base_html():
    html = et.Element('html')
    head = et.SubElement(html, 'head')
    meta = et.SubElement(head, 'meta')
    meta.set('charset', 'utf-8')
    meta = et.SubElement(head, 'meta')
    meta.set('name', 'viewport')
    meta.set('content', 'width=2000, initial-scale=1.0')
    body = et.SubElement(html, 'body')
    return html


def _table_html(results):
    log.info('Generating comparison table')

    def name_backend(result):
        return result['name'], result['backend']

    backends = set()
    names = set()
    classified = defaultdict(dict)
    for (name, backend), items in itertools.groupby(sorted(results,
                                                           key=name_backend),
                                                    key=name_backend):
        data = {item['float_type']: _classify(item['ci']) for item in items}
        classified[name][backend] = data['float'], data['double']
        names.add(name)
        backends.add(backend)

    backends = list(sorted(backends))
    names = list(sorted(names))

    table = et.Element('table')

    row = et.SubElement(table, 'tr')

    cell = et.SubElement(row, 'th')
    cell.text = 'BENCHMARK'
    for backend in backends:
        cell = et.SubElement(row, 'th')
        cell.text = backend.upper()

    for name in names:
        row = et.SubElement(table, 'tr')
        cell = et.SubElement(row, 'td')
        cell.text = name.replace('_', ' ').title()

        for backend in backends:
            try:
                float_class, double_class = classified[name][backend]
            except KeyError:
                cell = et.SubElement(row, 'td')
                continue
            if float_class == double_class:
                class_str = float_class
                r, g, b = _color(class_str)
            else:
                class_str = float_class + ' ' + double_class
                r, g, b = (_color(float_class) + _color(double_class)) / 2

            cell = et.SubElement(row, 'td')
            r, g, b = int(r), int(g), int(b)
            cell.set('style', f'background:rgb({r}, {g}, {b})')
            cell.text = class_str

    log.debug('Generated performance comparison table')
    return table


def _histogram_html(result):
    log.info('Plotting comparison histogram')
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=result['series_before'], name='Before'))
    fig.add_trace(go.Histogram(x=result['series_after'], name='After'))
    median_before = np.median(result['series_before'])
    median_after = np.median(result['series_after'])
    fig.add_shape(type='line',
                  x0=median_before,
                  x1=median_before,
                  y0=0,
                  y1=1,
                  yref='paper',
                  line=dict(color=colors.DEFAULT_PLOTLY_COLORS[0]))
    fig.add_shape(type='line',
                  x0=median_after,
                  x1=median_after,
                  y0=0,
                  y1=1,
                  yref='paper',
                  line=dict(color=colors.DEFAULT_PLOTLY_COLORS[1]))
    fig.update_layout(barmode='overlay',
                      font=dict(color='black'),
                      autosize=False,
                      width=990,
                      height=500)
    fig.update_traces(opacity=0.5)
    fig.update_xaxes(rangemode='tozero')
    log.debug('Generated histogram plot')
    return et.fromstring(fig.to_html(include_plotlyjs='cdn', full_html=False))


def _info_html(a, b):
    table = et.Element('table')

    def add_row(name, va, vb, tag='td'):
        row = et.SubElement(table, 'tr')
        cell = et.SubElement(row, tag)
        cell.text = name
        cell = et.SubElement(row, tag)
        cell.text = va
        cell = et.SubElement(row, tag)
        cell.text = vb
        return row

    add_row('Property', 'Before', 'After', tag='th')

    for k, vb in b['gridtools'].items():
        va = a['gridtools'].get(k, '')
        add_row('GridTools ' + k.title(), va, vb)

    for k, vb in b['environment'].items():
        va = a['environment'].get(k, '')
        add_row(k.title(), va, vb)

    log.debug('Generated info comparison table')
    return table


def compare(a, b, output):
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

    html = _base_html()

    style = et.SubElement(html.find('head'), 'style')
    style.text = (
        '''table { margin-bottom: 5em; border-collapse: collapse; }
           th { text-align: left;
                border-bottom: 1px solid black;
                padding: 0.5em; }
           td { padding: 0.5em; }
           .container { width: 100%;
                        display: flex;
                        flex-direction: row;
                        flex-wrap: wrap; }
           .item { width: 995px; }
           html { font-family: sans-serif; }''')

    body = html.find('body')

    title = et.SubElement(body, 'h1')
    title.text = 'GridTools Performance Test Results'

    p = et.SubElement(body, 'p')
    p.text = 'Domain size: ' + '×'.join(str(d) for d in b['domain'])

    title = et.SubElement(body, 'h2')
    title.text = 'Results'
    body.append(_table_html(results))

    title = et.SubElement(body, 'h2')
    title.text = 'Details'
    container = et.SubElement(body, 'div')
    container.set('class', 'container')

    for result in results:
        if _significant(result['ci']):
            item = et.SubElement(container, 'div')
            item.set('class', 'item')
            title = et.SubElement(item, 'h3')
            title.text = (result['name'].replace('_', ' ').title() + ' (' +
                          result['backend'].upper() + ', ' +
                          result['float_type'].upper() + ', ' +
                          _classify(result['ci']) + ')')
            item.append(_histogram_html(result))

    title = et.SubElement(body, 'h2')
    title.text = 'Info'
    body.append(_info_html(a, b))

    et.ElementTree(html).write(output, encoding='utf-8', method='html')
    log.info(f'Sucessfully written output to {output}')


def history(results, key='job', limit=None):
    """Plots run time history of all results. Depending on the argument `job`,
       The results are either ordered by commit/build time or job time
       (i.e. when the job was run).

    Args:
        results: List of `result.Result` objects.
        key: Either 'job' or 'build'.
        limit: Optionally limits the number of plotted results to the given
               number, i.e. only displays the most recent results. If `None`,
               all given results are plotted.
    """

    # get date/time either from the commit/build or job (when job was run)
    def get_datetime(result):
        if key == 'build':
            datetime = result.runinfo.datetime
        elif key == 'job':
            datetime = result.datetime
        else:
            raise ValueError('"key" argument must be "build" or "job"')
        return time.local_time(datetime)

    # sort results by desired reference time
    results = sorted(results, key=get_datetime)

    #
    if limit is not None:
        if not isinstance(limit, int) or limit <= 0:
            raise ValueError('"limit" must be a positive integer')
        results = results[-limit:]

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        percentiles = [
            result.times_by_stencil(results,
                                    missing=[np.nan],
                                    func=functools.partial(np.percentile, q=q))
            for q in (0, 25, 50, 75, 100)
        ]

    stencils = percentiles[0].keys()
    percentiles = {
        stencil: [p[stencil] for p in percentiles]
        for stencil in stencils
    }

    dates = [matplotlib.dates.date2num(get_datetime(r)) for r in results]

    if len(dates) > len(set(dates)):
        log.warning('Non-unique datetimes in history plot')

    fig, ax = plt.subplots(figsize=figsize(2, 1))

    locator = matplotlib.dates.AutoDateLocator()
    formatter = matplotlib.dates.AutoDateFormatter(locator)
    formatter.scaled[1 / 24] = '%y-%m-%d %H:%M'
    formatter.scaled[1 / (24 * 60)] = '%y-%m-%d %H:%M'
    formatter.scaled[1 / (24 * 60 * 60)] = '%y-%m-%d %H:%M:%S'

    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    colors = discrete_colors(len(stencils))

    for color, (stencil, qs) in zip(colors, sorted(percentiles.items())):
        mint, q1, q2, q3, maxt = qs
        ax.fill_between(dates, mint, maxt, alpha=0.2, color=color)
        ax.fill_between(dates, q1, q3, alpha=0.5, color=color)
        ax.plot(dates, q2, '|-', label=stencil.title(), color=color)

    ax.legend(loc='upper left')
    fig.autofmt_xdate()
    fig.tight_layout()

    return fig
