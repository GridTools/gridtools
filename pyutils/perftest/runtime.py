# -*- coding: utf-8 -*-

import abc
import os
import re
import subprocess

from perftest import NotFoundError, ParseError
from perftest import logger, result, runtools, stencils, time


class Runtime(metaclass=abc.ABCMeta):
    """Base class of all runtimes.

    A runtime class represents a software for running stencils, currently
    STELLA or Gridtools.
    """
    def __init__(self, config):
        from perftest import buildinfo

        self.config = config

        # Import build information
        self.grid = buildinfo.grid
        self.precision = buildinfo.precision
        self.backend = buildinfo.backend

        self.stencils = stencils.load(self.grid)

    def run(self, domain, runs, job_limit=None):
        """Method to run all stencils on the given `domain` size.

        Computes mean and stdev of the run times for all stencils for the
        given number of `runs`.

        Args:
            domain: Domain size as a tuple or list.
            runs: Number of runs to do.

        Returns:
            A `perftest.result.Result` object with the collected times.
        """
        # Get commands per stencil
        commands = [self.command(s, domain) for s in self.stencils]

        # Multiply commands by number of runs
        allcommands = [c for c in commands for _ in range(runs)]

        # Run commands
        logger.info('Running stencils')
        alloutputs = runtools.run(allcommands, self.config, job_limit)
        logger.info('Running stencils finished')

        # Parse outputs
        alltimes = [self._parse_time(o) for o in alloutputs]

        # Group times per stencil
        times = [alltimes[i:i+runs] for i in range(0, len(alltimes), runs)]

        return result.from_data(runtime=self,
                                domain=domain,
                                times=times)

    @staticmethod
    def _parse_time(output):
        """Parses the ouput of a STELLA or Gridtools run for the run time.

        Args:
            output: STELLA or Gridtools stdout output.

        Returns:
            Run time in seconds.
        """
        p = re.compile(r'.*\[s\]\s*([0-9.]+).*', re.MULTILINE | re.DOTALL)
        m = p.match(output)
        if not m:
            raise ParseError(f'Could not parse time in output:\n{output}')
        return float(m.group(1))

    def __str__(self):
        return (f'Runtime(name={self.name}, version={self.version}, '
                f'datetime={self.datetime})')

    @property
    def name(self):
        """Name, class name converted to lower case and 'Runtime' stripped."""
        return type(self).__name__.lower().rstrip('runtime')

    @abc.abstractproperty
    def version(self):
        """Version number or hash."""
        pass

    @abc.abstractproperty
    def datetime(self):
        """(Build or commit) date of the software as a string."""
        pass

    @abc.abstractproperty
    def compiler(self):
        pass

    @abc.abstractmethod
    def binary(self, stencil):
        """Stencil-dependent path to the binary."""
        pass

    @abc.abstractmethod
    def command(self, stencil, domain):
        """Full command to run the given stencil on the given dommain."""
        pass


class StellaRuntimeBase(Runtime):
    """Base class for all STELLA runtimes."""

    def binary(self, stencil):
        """STELLA binary path."""
        suffix = 'CUDA' if self.backend == 'cuda' else ''
        binary = os.path.join(self.path, f'StandaloneStencils{suffix}')
        if not os.path.isfile(binary):
            raise NotFoundError(f'Could not find STELLA binary at {binary}')
        return binary

    def command(self, stencil, domain):
        """STELLA run command."""
        binary = self.binary(stencil)
        ni, nj, nk = domain
        filt = stencil.stella_filter
        return f'{binary} --ie {ni} --je {nj} --ke {nk} --gtest_filter={filt}'

    @abc.abstractproperty
    def path(self):
        pass


class GridtoolsRuntime(Runtime):
    """Gridtools runtime class."""

    def __init__(self, config):
        super().__init__(config)
        from perftest import buildinfo

        # Import more gridtools-related build information
        self.source_dir = buildinfo.source_dir
        self.binary_dir = buildinfo.binary_dir
        self._compiler = buildinfo.compiler

        if buildinfo.build_type != 'release':
            logger.warning('You are running perftests with non-release build')

    @property
    def compiler(self):
        return self._compiler

    @property
    def version(self):
        """Gridtools git commit hash."""
        srcdir = self.source_dir
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                       cwd=srcdir).decode().strip()

    @property
    def datetime(self):
        """Gridtools git commit date and time."""
        commit = self.version
        srcdir = self.source_dir
        posixtime = subprocess.check_output(['git', 'show', '-s',
                                             '--format=%ct', commit],
                                            cwd=srcdir).decode().strip()
        return time.from_posix(posixtime)

    def binary(self, stencil):
        """Stencil-dependent Gridtools binary path."""
        binary = stencil.gridtools_binary(self.backend)
        binary = os.path.join(self.binary_dir, binary)
        if not os.path.isfile(binary):
            raise NotFoundError(f'Could not find GridTools binary at {binary}')
        return binary

    def command(self, stencil, domain):
        """Gridtools run command."""
        binary = self.binary(stencil)
        ni, nj, nk = domain
        halo = stencil.halo
        ni, nj = ni + 2 * halo, nj + 2 * halo
        return f'{binary} {ni} {nj} {nk} 10'
