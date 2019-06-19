# -*- coding: utf-8 -*-

import collections
import json

from pyutils import log
from perftest import time


version = 0.5


Result = collections.namedtuple('Result',
                                ['version', 'datetime', 'runinfo', 'domain',
                                 'times'])


RunInfo = collections.namedtuple('RunInfo',
                                 ['name', 'version', 'datetime', 'precision',
                                  'backend', 'grid', 'compiler', 'hostname',
                                  'clustername'])


Time = collections.namedtuple('Time',
                              ['stencil', 'measurements'])


def from_data(runinfo, domain, stencils, times):
    """Creates a Result object from collected data.

    Args:
        runinfo: A `RunInfo` object.
        domain: The domain size as a tuple or list.
        times: List of list of run times per stencil.
    """
    times_data = [Time(stencil=s.name, measurements=t) for s, t
                  in zip(stencils, times)]

    return Result(runinfo=runinfo,
                  times=times_data,
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
        if isinstance(d, time.datetime):
            return time.timestr(d)
        try:
            return d._asdict()
        except AttributeError:
            return d

    with open(filename, 'w') as fp:
        json.dump(data, fp, indent=4, sort_keys=True, default=convert)
    log.info(f'Successfully saved result to {filename}')


def load(filename):
    """Loads result data from the given json file.

    Args:
        filename: The name of the input file.
    """
    with open(filename, 'r') as fp:
        data = json.load(fp)

    if data['version'] == version:
        d = data['runinfo']
        runinfo_data = RunInfo(name=d['name'],
                               version=d['version'],
                               datetime=time.from_timestr(d['datetime']),
                               precision=d['precision'],
                               backend=d['backend'],
                               grid=d['grid'],
                               compiler=d['compiler'],
                               hostname=d['hostname'],
                               clustername=d['clustername'])
    elif data['version'] == 0.4:
        runtime_data = data['runtime']
        config_data = data['config']
        runinfo_data = RunInfo(name=runtime_data['name'],
                               version=runtime_data['version'],
                               datetime=time.from_timestr(
                                   runtime_data['datetime']),
                               precision=runtime_data['precision'],
                               backend=runtime_data['backend'],
                               grid=runtime_data['grid'],
                               compiler=runtime_data['compiler'],
                               hostname=config_data['hostname'],
                               clustername=config_data['clustername'])
    elif data['version'] != version:
        raise ValueError(f'Unknown result file version "{data["version"]}"')

    times_data = [Time(stencil=d['stencil'], measurements=d['measurements'])
                  for d in data['times']]

    result = Result(runinfo=runinfo_data,
                    times=times_data,
                    domain=data['domain'],
                    datetime=time.from_timestr(data['datetime']),
                    version=data['version'])
    log.info(f'Successfully loaded result from {filename}')
    return result


def times_by_stencil(results, *, func=None, missing=None):
    """Returns a dictionary, mapping stencil names to lists of measured times.
    Optionally applies `func` to each list of measurments.

    Args:
        result: A ist of `Result` objects.
        func: Function to apply to each list of measurements.
        missing: Value that replaces missing data values (only used if
                 multiple results are given).

    Returns:
        A dictionary mapping stencil names to measured times (optionally
        transformed by func).
    """
    if func is None:
        def identity(x):
            return x
        func = identity

    times = [{t.stencil: func(t.measurements) for t in r.times}
             for r in results]
    stencils = set.union(*(set(t.keys()) for t in times))

    return {stencil: [t.get(stencil, func(missing)) for t in times]
            for stencil in stencils}


def compare(results):
    """Compares multiple results and splits equal and unequal parts.

    Args:
        results: A list of `Result` objects.

    Returns:
        A tuple of one `Data` object holding all common result attributes and
        a list of `Data` objects holding all other (unequal) attributes.
    """
    first, *rest = [r._asdict() for r in results]
    common_keys = set(first.keys()).intersection(*(r.keys() for r in rest))
    common = {k: first[k] for k in common_keys
              if all(first[k] == r[k] for r in rest)}
    diff = [{k: v for k, v in r._asdict().items() if k not in common.keys()}
            for r in results]
    return common, diff
