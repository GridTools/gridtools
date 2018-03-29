# -*- coding: utf-8 -*-

import json

from perftest import ArgumentError, logger, ParseError, time


version = 0.3


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

    @property
    def meantimes(self):
        """List of all stencils' mean computation time."""
        return [t.mean for t in self.times]

    @property
    def stdevtimes(self):
        """List of all stencils' computation time stdandard deviation."""
        return [t.stdev for t in self.times]


def from_data(runtime, domain, meantimes, stdevtimes, runs):
    """Creates a Result object from collected data.

    Args:
        runtime: A `perftest.runtime.Runtime` object.
        domain: The domain size as a tuple or list.
        meantimes: List of mean run times per stencil.
        stdevtimes: List of stdev run times perf stencil.
    """
    times_data = [Data(stencil=s.name, mean=m, stdev=d, runs=runs) for s, m, d
                  in zip(runtime.stencils, meantimes, stdevtimes)]

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

    times_data = [Data(**d) for d in data['times']]
    runtime_data = Data(**data['runtime'])
    config_data = Data(**data['config'])

    times_data = [Data(stencil=d['stencil'], mean=d['mean'], stdev=d['stdev'],
                       runs=d['runs'])
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


def times_by_stencil(results):
    """Computes mean and stdev times of multiple results by stencil.

    Args:
        results: List of `Result` objects.

    Returns:
        A tuple of lists (stencils, meantimes, stdevtimes).
    """
    stencils = results[0].stencils
    if any(stencils != r.stencils for r in results):
        raise ArgumentError('All results must include the same stencils')

    meantimes = list(zip(*(r.meantimes for r in results)))
    stdevtimes = list(zip(*(r.stdevtimes for r in results)))
    return stencils, meantimes, stdevtimes


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
