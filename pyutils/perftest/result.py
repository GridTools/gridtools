# -*- coding: utf-8 -*-

import json

import numpy as np

from perftest import ArgumentError, logger, ParseError, time


version = 0.4


class Data(dict):
    """Base class for all result data.

    This class derives from `dict`, but overrides __getattr__ to additionally
    allow item access with attribute syntax.
    """

    def __getattr__(self, name):
        """Redirects accesses to unknown attributes to item accesses."""
        return self[name]

    def __repr__(self):
        return 'Data(' + ', '.join(f'{k}={v!r}' for k, v in self.items()) + ')'


class Result(Data):
    """Class for storing the result data of a run."""

    @property
    def stencils(self):
        """List of all stencils (names of stencils)."""
        return [t.stencil for t in self.times]

    def times_by_stencil(self):
        """List of all timing, grouped by stencil."""
        return [t.measurements for t in self.times]

    def mapped_times(self, func):
        return [func(t.measurements) for t in self.times]


def from_data(runtime, domain, times):
    """Creates a Result object from collected data.

    Args:
        runtime: A `perftest.runtime.Runtime` object.
        domain: The domain size as a tuple or list.
        times: List of list of run times per stencil.
    """
    times_data = [Data(stencil=s.name, measurements=t) for s, t
                  in zip(runtime.stencils, times)]

    runtime_data = Data(name=runtime.name,
                        version=runtime.version,
                        datetime=runtime.datetime,
                        grid=runtime.grid,
                        precision=runtime.precision,
                        backend=runtime.backend,
                        compiler=runtime.compiler)

    config_data = Data(configname=runtime.config.name,
                       hostname=runtime.config.hostname,
                       clustername=runtime.config.clustername)

    return Result(runtime=runtime_data,
                  times=times_data,
                  config=config_data,
                  domain=domain,
                  datetime=time.now(),
                  version=version)


def save(filename, data):
    """Saves the result data to the given a json file.

    Overwrites the file if it already exists.

    Args:
        filename: The name of the output file.
        data: An instance of `Result`.
    """
    def convert(d):
        if isinstance(d, Data):
            return {k: v for k, v in d.items()}
        elif isinstance(d, time.datetime):
            return time.timestr(d)

    with open(filename, 'w') as fp:
        json.dump(data, fp, indent=4, sort_keys=True, default=convert)
    logger.info(f'Successfully saved result to {filename}')


def load(filename):
    """Loads result data from the given json file.

    Args:
        filename: The name of the input file.
    """
    with open(filename, 'r') as fp:
        data = json.load(fp)

    if data['version'] != version:
        raise ParseError('Unknown result file version')

    times_data = [Data(stencil=d['stencil'], measurements=d['measurements'])
                  for d in data['times']]

    d = data['runtime']
    runtime_data = Data(name=d['name'],
                        version=d['version'],
                        datetime=time.from_timestr(d['datetime']),
                        grid=d['grid'],
                        precision=d['precision'],
                        backend=d['backend'],
                        compiler=d['compiler'])

    d = data['config']
    config_data = Data(configname=d['configname'],
                       hostname=d['hostname'],
                       clustername=d['clustername'])

    result = Result(runtime=runtime_data,
                    times=times_data,
                    config=config_data,
                    domain=data['domain'],
                    datetime=time.from_timestr(data['datetime']),
                    version=data['version'])
    logger.info(f'Successfully loaded result from {filename}')
    return result


def by_stencils(results_data):
    return list(zip(*results_data))


def times_by_stencil(results):
    """Collects times of multiple results by stencils.

    Args:
        results: List of `Result` objects.

    Returns:
        A tuple of lists (stencils, times).
    """
    stencils = results[0].stencils
    if any(stencils != r.stencils for r in results):
        raise ArgumentError('All results must include the same stencils')

    times = by_stencils(r.times_by_stencil() for r in results)
    return stencils, times


def statistics_by_stencil(results):
    """Computes mean and stdev times of multiple results by stencil.

    Args:
        results: List of `Result` objects.

    Returns:
        A tuple of lists (stencils, meantimes, stdevtimes).
    """
    stencils = results[0].stencils
    if any(stencils != r.stencils for r in results):
        raise ArgumentError('All results must include the same stencils')

    meantimes = by_stencils(r.mapped_times(np.mean) for r in results)
    stdevtimes = by_stencils(r.mapped_times(np.std) for r in results)
    return stencils, meantimes, stdevtimes


def percentiles_by_stencil(results, percentiles):
    stencils = results[0].stencils
    if any(stencils != r.stencils for r in results):
        raise ArgumentError('All results must include the same stencils')

    qtimes = []
    for q in percentiles:
        def compute_q(times):
            return np.percentile(times, q)
        qtimes.append(by_stencils(r.mapped_times(compute_q) for r in results))

    return (stencils, *qtimes)


def compare(results):
    """Compares multiple results and splits equal and unequal parts.

    Args:
        results: A list of `Result` objects.

    Returns:
        A tuple of one `Data` object holding all common result attributes and
        a list of `Data` objects holding all other (unequal) attributes.
    """
    first, *rest = results
    common_keys = set(first.keys()).intersection(*(r.keys() for r in rest))
    common = {k: first[k] for k in common_keys
              if all(first[k] == r[k] for r in rest)}
    diff = [{k: v for k, v in r.items() if k not in common.keys()}
            for r in results]
    return Data(common), [Data(d) for d in diff]
