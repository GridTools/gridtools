# -*- coding: utf-8 -*-

from collections import defaultdict
import contextlib
import functools
import itertools
import math
import operator
import os
import re
import sys
import types
import warnings

import numpy as np

from pyutils import log

from plotly import graph_objs as go, colors


def figsize(rows=1, cols=1):
    """Default figsize for a plot with given subplot rows and columns."""
    return (7 * rows, 5 * cols)


def discrete_colors(n):
    """Generates `n` discrete colors for plotting."""
    colors = plt.cm.tab20.colors
    colors = colors[::2] + colors[1::2]
    return list(itertools.islice(itertools.cycle(colors), n))


def get_titles(results):
    """Generates plot titles. Compares the stored run info in the results
    and groups them by common and different values to automtically generate
    figure title and plot captions.

    Args:
        results: List of `result.Result` objects.

    Returns:
        A tuple with two elements. The first element is a single string,
        usable as a common title. The second element is a list of strings,
        one per given result, usable as subtitles or plot captions.
    """
    common, diff = result.compare([r.runinfo for r in results])

    def titlestr(k, v):
        if k == 'name':
            return 'Runtime: ' + v.title()
        if k == 'datetime':
            return 'Date/Time: ' + time.short_timestr(time.local_time(v))
        elif k == 'compiler':
            m = re.match(
                '(?P<compiler>[^ ]+) (?P<version>[^ ]+)'
                '( \\((?P<compiler2>[^ ]+) (?P<version2>[^ ]+)\\))?', v)
            if m:
                d = m.groupdict()
                d['compiler'] = os.path.basename(d['compiler'])
                v = '{compiler} {version}'.format(**d)
                if d['compiler2']:
                    d['compiler2'] = os.path.basename(d['compiler2'])
                    v += ' ({compiler2} {version2})'.format(**d)
            return 'Compiler: ' + v
        else:
            s = str(v).title()
            if len(s) > 20:
                s = s[:20] + '…'
            return k.title() + ': ' + s

    suptitle = ', '.join(titlestr(k, v) for k, v in common.items())

    titles = ['\n'.join(titlestr(k, v) for k, v in d.items()) for d in diff]
    return suptitle, titles


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


@contextlib.contextmanager
def _html(filename, style=None):
    with open(filename, 'w') as file:
        file.write('<html>\n')
        file.write('<head>\n')
        file.write('<meta charset="UTF-8">\n')
        file.write('<meta name="viewport" '
                   'content="width=1000, initial-scale=1.0">\n')
        if style is not None:
            file.write('<style>html { font-family: sans-serif; } ' + style +
                       '</style>\n')
        file.write('</head>\n')
        file.write('<body>\n')
        yield file
        file.write('</body>\n')
        file.write('</html>\n')
    log.info(f'Successfully written {filename}')


def _write_table_html(results, html):
    log.info('Writing comparison table')
    with _html(html, style='td { padding: 0.5em; }') as file:

        def name_backend(result):
            return result['name'], result['backend']

        backends = set()
        names = set()
        classified = defaultdict(dict)
        for (name,
             backend), items in itertools.groupby(sorted(results,
                                                         key=name_backend),
                                                  key=name_backend):
            data = {
                item['float_type']: _classify(item['ci'])
                for item in items
            }
            classified[name][backend] = data['float'], data['double']
            names.add(name)
            backends.add(backend)

        backends = list(sorted(backends))
        names = list(sorted(names))

        file.write('<table>\n<tr>\n')
        file.write('    <td style="font-weight:bold">BENCHMARK</td>\n')
        for backend in backends:
            file.write(
                f'    <td style="font-weight:bold">{backend.upper()}</td>\n')
        file.write('  </tr>\n')

        for name in names:
            file.write('  <tr>\n')
            title = name.replace('_', ' ').title()
            file.write(f'    <td>{title}</td>\n')
            for backend in backends:
                try:
                    float_class, double_class = classified[name][backend]
                except KeyError:
                    file.write('    <td></td>\n')
                    continue
                if float_class == double_class:
                    class_str = float_class
                    r, g, b = _color(class_str)
                else:
                    class_str = float_class + ' ' + double_class
                    r, g, b = (_color(float_class) + _color(double_class)) / 2

                file.write(
                    f'    <td style="background:rgb({r}, {g}, {b})">{class_str}</td>\n'
                )
            file.write('  </tr>\n')
        file.write('</table>\n')


def _write_histogram_html(result, html):
    log.info('Plotting comparison histogram')
    title = (result['name'].replace('_', ' ').title() + ' (' +
             result['backend'].upper() + ', ' + result['float_type'].upper() +
             ', ' + _classify(result['ci']) + ')')
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
    fig.update_layout(title=title, barmode='overlay', font=dict(color='black'))
    fig.update_traces(opacity=0.5)
    fig.update_xaxes(rangemode='tozero')
    fig.write_html(html)


def _write_combined_html(iframes, html):
    log.debug('Generating combined HTML', ', '.join(iframes) + ' -> ' + html)
    with _html(html, style='iframe { width: 1000px; height: 500px; }') as file:
        for iframe in iframes:
            file.write(f'<iframe src="{iframe}" frameborder=0 scrolling="no">'
                       f'</iframe>')


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

    base, ext = os.path.splitext(output)
    htmls = [f'{base}_table{ext}']
    _write_table_html(results, htmls[-1])

    for result in results:
        if _significant(result['ci']):
            html = f'{base}_plot_{len(htmls)}{ext}'
            _write_histogram_html(result, html)
            htmls.append(html)

    _write_combined_html(htmls, output)


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
