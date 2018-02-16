# -*- coding: utf-8 -*-

import json

from perftest import logger, utils


class RuntimeInfo:
    def __init__(self, name, version, datetime, grid, precision, backend):
        self.name = name
        self.version = version
        self.datetime = utils.datetime_from_timestr(datetime)
        self.grid = grid
        self.precision = precision
        self.backend = backend

    def compare(self, *others):
        common = dict()
        diff = dict()
        attrs = ['name', 'version', 'datetime', 'grid', 'precision', 'backend']
        for attr in attrs:
            if all(getattr(self, attr) == getattr(o, attr) for o in others):
                common[attr] = getattr(self, attr)
            else:
                diff[attr] = [getattr(self, attr)] + [getattr(o, attr) for o
                                                      in others]
        return common, diff


class Result:
    def __init__(self, filename=None, runtime=None, domain=None,
                 stencils=None, meantimes=None, stdevtimes=None):
        if filename is None:
            self._init_from_run(runtime, domain, stencils,
                                meantimes, stdevtimes)
        else:
            self._init_from_file(filename)

    def write(self, filename):
        with open(filename, 'w') as fp:
            json.dump(self.data, fp, indent=4, sort_keys=True)
        logger.info(f'Wrote result to {filename}')

    def _init_from_file(self, filename):
        with open(filename, 'r') as fp:
            self.data = json.load(fp)
        logger.info(f'Successfully loaded result from {filename}')

    def _init_from_run(self, runtime, domain, stencils, meantimes, stdevtimes):
        if None in (runtime, domain, stencils, meantimes, stdevtimes):
            raise ArgumentError('Invalid arguments')

        times = [{'stencil': s.name, 'mean': m, 'stdev': d} for s, m, d in
                 zip(stencils, meantimes, stdevtimes)]

        self.data = {'runtime': {'name': runtime.name,
                                 'version': runtime.version,
                                 'datetime': runtime.datetime,
                                 'grid': runtime.grid,
                                 'precision': runtime.precision,
                                 'backend': runtime.backend},
                     'domain': domain,
                     'times': times,
                     'datetime': utils.timestr()}

    @property
    def stencils(self):
        return [t['stencil'] for t in self.data['times']]

    @property
    def meantimes(self):
        return [t['mean'] for t in self.data['times']]

    @property
    def stdevtimes(self):
        return [t['stdev'] for t in self.data['times']]

    @property
    def domain(self):
        return self.data['domain']

    @property
    def runtime(self):
        return RuntimeInfo(**self.data['runtime'])

    @property
    def datetime(self):
        return utils.datetime_from_timestr(self.data['datetime'])

    def times_by_stencil(self, *others):
        if not all(self.stencils == o.stencils for o in others):
            raise ArgumentError('All results must include the same stencils')
        combined = (self,) + others
        mtimes = list(zip(*(r.meantimes for r in combined)))
        stimes = list(zip(*(r.stdevtimes for r in combined)))
        return self.stencils, mtimes, stimes


def times_by_stencil(*results):
    return results[0].times_by_stencil(*results[1:])
