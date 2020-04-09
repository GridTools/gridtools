# -*- coding: utf-8 -*-

import typing

import dateutil.parser
import matplotlib
from matplotlib import pyplot as plt
import numpy as np

from pyutils import log
from perftest import html

plt.style.use('ggplot')


class _OutputKey(typing.NamedTuple):
    name: str
    backend: str
    float_type: str

    def __str__(self):
        name = self.name.replace('_', ' ').title()
        backend = self.backend.upper()
        float_type = self.float_type
        return f'{name} ({backend}, {float_type})'

    @classmethod
    def outputs_by_key(cls, data):
        def split_output(o):
            return cls(**{k: v
                          for k, v in o.items() if k != 'series'}), o['series']

        return dict(split_output(o) for o in data['outputs'])


class _ConfidenceInterval(typing.NamedTuple):
    lower: float
    upper: float

    def classify(self):
        assert self.lower <= self.upper

        # large uncertainty
        if self.upper - self.lower > 0.1:
            return '??'

        # no change
        if -0.01 <= self.lower <= 0 <= self.upper <= 0.01:
            return '='
        if -0.02 <= self.lower <= self.upper <= 0.02:
            return '(=)'

        # probably no change, but quite large uncertainty
        if -0.05 <= self.lower <= 0 <= self.upper <= 0.05:
            return '?'

        # faster
        if -0.01 <= self.lower <= 0.0:
            return '(+)'
        if -0.05 <= self.lower <= -0.01:
            return '+'
        if -0.1 <= self.lower <= -0.05:
            return '++'
        if self.upper <= -0.1:
            return '+++'

        # slower
        if 0.01 >= self.upper >= 0.0:
            return '(-)'
        if 0.05 >= self.upper >= 0.01:
            return '-'
        if 0.1 >= self.upper >= 0.05:
            return '--'
        if self.lower >= 0.1:
            return '---'

        # no idea
        return '???'

    def significant(self):
        return '=' not in self.classify()

    def __str__(self):
        assert self.lower <= self.upper
        plower, pupper = 100 * self.lower, 100 * self.upper

        if self.lower <= 0 and self.upper <= 0:
            return f'{-pupper:3.1f}% – {-plower:3.1f}% faster'
        if self.lower >= 0 and self.upper >= 0:
            return f'{plower:3.1f}% – {pupper:3.1f}% slower'
        return f'{-plower:3.1f}% faster – {pupper:3.1f}% slower'

    @classmethod
    def compare_medians(cls, before, after, n=1000, alpha=0.05):
        scale = np.median(before)
        before = np.asarray(before) / scale
        after = np.asarray(after) / scale
        # bootstrap sampling
        before_samples = np.random.choice(before, (before.size, n))
        after_samples = np.random.choice(after, (after.size, n))
        # bootstrap estimates of difference of medians
        bootstrap_estimates = (np.median(after_samples, axis=0) -
                               np.median(before_samples, axis=0))
        # percentile bootstrap confidence interval
        ci = np.quantile(bootstrap_estimates, [alpha / 2, 1 - alpha / 2])
        log.debug(f'Boostrap results (n = {n}, alpha = {alpha})',
                  f'{ci[0]:8.5f} - {ci[1]:8.5f}')
        return cls(*ci)


def _add_comparison_table(report, cis):
    names = list(sorted(set(k.name for k in cis.keys())))
    backends = list(sorted(set(k.backend for k in cis.keys())))

    def css_class(classification):
        if '-' in classification:
            return 'bad'
        if '?' in classification:
            return 'unknown'
        if '+' in classification:
            return 'good'
        return ''

    with report.table('Comparison') as table:
        with table.row() as row:
            row.fill('BENCHMARK', *(b.upper() for b in backends))

        for name in names:
            with table.row() as row:
                name_cell = row.cell(name.replace('_', ' ').title())
                row_classification = ''
                for backend in backends:
                    try:
                        classification = [
                            cis[_OutputKey(name=name,
                                           backend=backend,
                                           float_type=float_type)].classify()
                            for float_type in ('float', 'double')
                        ]
                        if classification[0] == classification[1]:
                            classification = classification[0]
                        else:
                            classification = ' '.join(classification)
                    except KeyError:
                        classification = ''
                    row_classification += classification
                    row.cell(classification).set('class',
                                                 css_class(classification))
                name_cell.set('class', css_class(row_classification))

    with report.table('Explanation of Symbols') as table:
        def add_help(string, meaning):
            with table.row() as row:
                row.fill(string, meaning)

        add_help('Symbol', 'MEANING')
        add_help('=', 'No performance change (confidence interval within ±1%)')
        add_help(
            '(=)',
            'Probably no performance change (confidence interval within ±2%)')
        add_help('(+)/(-)',
                 'Very small performance improvement/degradation (≤1%)')
        add_help('+/-', 'Small performance improvement/degradation (≤5%)')
        add_help('++/--', 'Large performance improvement/degradation (≤10%)')
        add_help('+++/---',
                 'Very large performance improvement/degradation (>10%)')
        add_help(
            '?', 'Probably no change, but quite large uncertainty '
            '(confidence interval with ±5%)')
        add_help('??', 'Unclear result, very large uncertainty (±10%)')
        add_help('???', 'Something unexpected…')

    log.debug('Generated performance comparison table')


def _histogram_plot(title, before, after, output):
    fig, ax = plt.subplots(figsize=(10, 5))
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


def _add_comparison_plots(report, before_outs, after_outs, cis):
    with report.image_grid('Details') as grid:
        for k, ci in cis.items():
            if ci.significant():
                title = (str(k) + ': ' + str(ci))
                _histogram_plot(title, before_outs[k], after_outs[k],
                                grid.image())


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


def compare(before, after, output):
    before_outs = _OutputKey.outputs_by_key(before)
    after_outs = _OutputKey.outputs_by_key(after)
    cis = {
        k: _ConfidenceInterval.compare_medians(before_outs[k], v)
        for k, v in after_outs.items() if k in before_outs
    }

    with html.Report(output, 'GridTools Performance Test Results') as report:
        _add_comparison_table(report, cis)
        _add_comparison_plots(report, before_outs, after_outs, cis)
        _add_comparison_info(report, before, after)


class _Measurements(typing.NamedTuple):
    min: list
    q1: list
    q2: list
    q3: list
    max: list

    def append(self, *values):
        assert len(self) == len(values)
        for l, v in zip(self, values):
            l.append(v)


def _history_data(data, key, limit):
    def get_datetime(result):
        source = 'gridtools' if key == 'commit' else 'environment'
        return dateutil.parser.isoparse(result[source]['datetime'])

    data = sorted(data, key=get_datetime)
    if limit:
        data = data[-limit:]

    datetimes = [get_datetime(d) for d in data]
    outputs = [_OutputKey.outputs_by_key(d) for d in data]

    keys = set.union(*(set(o.keys()) for o in outputs))
    measurements = {k: _Measurements([], [], [], [], []) for k in keys}
    for o in outputs:
        for k in keys:
            try:
                data = np.percentile(o[k], [0, 25, 50, 75, 100])
            except KeyError:
                data = [np.nan] * 5
            measurements[k].append(*data)

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

    style = next(iter(plt.rcParams['axes.prop_cycle']))
    ax.fill_between(dates,
                    measurements.min,
                    measurements.max,
                    alpha=0.2,
                    **style)
    ax.fill_between(dates,
                    measurements.q1,
                    measurements.q3,
                    alpha=0.5,
                    **style)
    ax.plot(dates, measurements.q2, '|-', **style)
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
            for k, m in sorted(measurements.items()):
                _history_plot(str(k), dates, m, grid.image())
