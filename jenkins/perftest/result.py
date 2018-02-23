# -*- coding: utf-8 -*-

import json

from perftest import ArgumentError, logger, utils


def _items_as_attrs(x):
    istype = isinstance(x, type)

    class Wrapper(x if istype else type(x)):
        def __getattr__(self, name):
            try:
                return getattr(super(), name)
            except AttributeError:
                return _items_as_attrs(self.__getitem__(name))

        def __getitem__(self, name):
            return _items_as_attrs(super().__getitem__(name))

    return Wrapper if istype else Wrapper(x)


@_items_as_attrs
class Result(dict):
    def __init__(self, filename=None, runtime=None, domain=None,
                 meantimes=None, stdevtimes=None):
        if filename:
            self._init_from_file(filename)
        else:
            self._init_from_run(runtime, domain, meantimes, stdevtimes)

    def _init_from_file(self, filename):
        with open(filename, 'r') as fp:
            self.update(json.load(fp))
        logger.info(f'Successfully loaded result from {filename}')

    def _init_from_run(self, runtime, domain, meantimes, stdevtimes):
        if None in (runtime, domain, meantimes, stdevtimes):
            raise ArgumentError('Invalid arguments')

        times = [{'stencil': s.name, 'mean': m, 'stdev': d} for s, m, d in
                 zip(runtime.stencils, meantimes, stdevtimes)]

        self.update({'runtime': {'name': runtime.name,
                                 'version': runtime.version,
                                 'datetime': runtime.datetime,
                                 'grid': runtime.grid,
                                 'precision': runtime.precision,
                                 'backend': runtime.backend},
                     'domain': domain,
                     'times': times,
                     'datetime': utils.timestr(),
                     'config': {
                         'configname': runtime.config.name,
                         'hostname': runtime.config.hostname,
                         'systemname': runtime.config.systemname,
                     }})

    def write(self, filename):
        with open(filename, 'w') as fp:
            json.dump(self, fp, indent=4, sort_keys=True)
        logger.info(f'Wrote result to "{filename}"')

    @property
    def stencils(self):
        return [t['stencil'] for t in self['times']]

    @property
    def meantimes(self):
        return [t['mean'] for t in self['times']]

    @property
    def stdevtimes(self):
        return [t['stdev'] for t in self['times']]


def times_by_stencil(results):
    stencils = results[0].stencils
    if any(stencils != r.stencils for r in results):
        raise ArgumentError('All results must include the same stencils')

    meantimes = list(zip(*(r.meantimes for r in results)))
    stdevtimes = list(zip(*(r.stdevtimes for r in results)))
    return stencils, meantimes, stdevtimes


def compare(results):
    first, *rest = results
    common_keys = set(first.keys()).intersection(*(r.keys() for r in rest))
    common = {k: first[k] for k in common_keys
              if all(first[k] == r[k] for r in rest)}
    diff = [{k: v for k, v in r.items() if k not in common.keys()}
            for r in results]
    return _items_as_attrs(common), _items_as_attrs(diff)
