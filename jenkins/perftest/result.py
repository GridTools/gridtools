# -*- coding: utf-8 -*-

import json

from perftest import ArgumentError, logger, utils


class Data(dict):
    def __getattr__(self, name):
        return self[name]


class Result(Data):
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


def from_data(filename, runtime, domain, meantimes, stdevtimes):
    """Creates a Data object from collected data.

    Args:
        runtime: A `perftest.runtime.Runtime` object.
        domain: The domain size as a tuple or list.
        meantimes: List of mean run times per stencil.
        stdevtimes: List of stdev run times perf stencil.
    """

    times_data = [Data(stencil=s.name, mean=m, stdev=d) for s, m, d in
                  zip(runtime.stencils, meantimes, stdevtimes)]

    runtime_data = Data(name=runtime.name,
                        version=runtime.version,
                        datetime=runtime.datetime,
                        grid=runtime.grid,
                        precision=runtime.precision,
                        backend=runtime.backend)

    config_data = Data(configname=runtime.config.name,
                       hostname=runtime.config.hostname,
                       systemname=runtime.config.systemname)

    return Result(runtime=runtime_data,
                  times=times_data,
                  config=config_data,
                  domain=domain,
                  datetime=utils.timestr())


def save(filename, data):
    def convert(d):
        return {k: v for k, v in d.items()}

    with open(filename, 'w') as fp:
        json.dump(data, fp, indent=4, sort_keys=True, default=convert)
    logger.info(f'Successfully saved result to {filename}')



def load(filename):
    """Loads result data from the given file."""
    with open(filename, 'r') as fp:
        data = json.load(fp)

    times_data = [Data(**d) for d in data['times']]
    runtime_data = Data(**data['runtime'])
    config_data = Data(**data['config'])

    result = Result(runtime=runtime_data,
                    times=times_data,
                    config=config_data,
                    domain=data['domain'],
                    datetime=data['datetime'])
    logger.info(f'Successfully loaded result from {filename}')
    return resut


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
        A tuple of one object holding all common result attributes and a list
        of objects holding all other (unequal) attributes.
    """
    first, *rest = results
    common_keys = set(first.keys()).intersection(*(r.keys() for r in rest))
    common = {k: first[k] for k in common_keys
              if all(first[k] == r[k] for r in rest)}
    diff = [{k: v for k, v in r.items() if k not in common.keys()}
            for r in results]
    return _items_as_attrs(common), _items_as_attrs(diff)
