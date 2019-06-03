#!/usr/bin/env python3

import os

from pyutils import args, env, log


script_dir = os.path.dirname(os.path.abspath(__file__))


@args.command(description='main script for GridTools pyutils')
@args.arg('--verbose', '-v', action='count', default=0,
          help='increase verbosity (use -vv for all debug messages)')
@args.arg('--logfile', '-l', help='path to logfile')
def driver(verbose, logfile):
    log.set_verbosity(verbose)
    if logfile:
        log.log_to_file(logfile)


@driver.command(description='build GridTools')
@args.arg('--build-type', '-b', choices=['release', 'debug'], required=True)
@args.arg('--precision', '-p', choices=['float', 'double'], required=True)
@args.arg('--grid', '-g', choices=['structured', 'icosahedral'],
          default='structured')
@args.arg('--environment', '-e', help='path to environment file')
@args.arg('--target', '-t', nargs='+', help='make targets to build')
@args.arg('--source-dir', help='GridTools source directory')
@args.arg('--build-dir', '-o', required=True, help='build directory')
@args.arg('--install-dir', '-i', help='install directory')
@args.arg('--cmake-only', action='store_true',
          help='only execute CMake but do not build')
def build(build_type, precision, grid, environment, target, source_dir,
          build_dir, install_dir, cmake_only):
    import build

    if source_dir is None:
        source_dir = os.path.abspath(os.path.join(script_dir, os.path.pardir))

    env.set_cmake_arg('CMAKE_BUILD_TYPE', build_type.title())
    env.set_cmake_arg('GT_SINGLE_PRECISION', precision == 'float')
    env.set_cmake_arg('GT_TESTS_ICOSAHEDRAL_GRID', grid == 'icosahedral')

    if environment:
        env.load(environment)

    build.cmake(source_dir, build_dir, install_dir)
    if not cmake_only:
        build.make(build_dir, target)


try:
    from pyutils import buildinfo
except ImportError:
    buildinfo = None


if buildinfo:
    @driver.command(description='run GridTools tests')
    @args.arg('--run-mpi-tests', '-m', action='store_true',
              help='enable execution of MPI tests')
    @args.arg('--perftests-only', action='store_true',
              help='only run perftests binaries')
    @args.arg('--verbose-ctest', action='store_true',
              help='run ctest in verbose mode')
    @args.arg('--examples-build-dir', help='build directory for examples')
    @args.arg('--build-examples', '-b', action='store_true',
              help='enable building of GridTools examples')
    def test(run_mpi_tests, perftests_only, verbose_ctest,
             examples_build_dir, build_examples):
        import test

        if not examples_build_dir:
            examples_build_dir = os.path.join(buildinfo.binary_dir,
                                              'examples_build')
        if perftests_only:
            label = 'perftests_*'
        else:
            label = '(unittest_*|regression_*)'
            
        if run_mpi_tests:
            mpi_label = 'mpitest_*'
        else:
            mpi_label = None
            
        test.run(label, mpi_label, verbose_ctest)
        if build_examples:
            test.compile_examples(examples_build_dir)


@driver.command(description='performance test utilities')
def perftest():
    pass


if buildinfo:
    @perftest.command(description='run performance tests')
    @args.arg('--domain-size', '-s', required=True, type=int, nargs=3,
              metavar=('ISIZE', 'JSIZE', 'KSIZE'),
              help='domain size (excluding halo)')
    @args.arg('--runs', default=10, type=int,
              help='number of runs to do for each stencil')
    @args.arg('--output', '-o', required=True,
              help='output file path, extension .json is added if not given')
    def run(domain_size, runs, output):
        import perftest
        if not output.lower().endswith('.json'):
            output += '.json'

        results = perftest.run(domain_size, runs)
        for tag, result in results.items():
            perftest.result.save(f'.{tag}.'.join(output.rsplit('.', 1)),
                                 result)


@perftest.command(description='plot performance results')
def plot():
    pass


@plot.command(description='plot performance comparison')
@args.arg('--output', '-o', required=True,
          help='output file, can have any extension supported by matplotlib')
@args.arg('--input', '-i', required=True, nargs='+',
          help='any number of input files')
def compare(output, input):
    from perftest import plot, result
    results = [result.load(f) for f in input]
    plot.compare(results).savefig(output)
    log.info(f'Successfully saved plot to {output}')


@plot.command(description='plot performance history')
@args.arg('--output', '-o', required=True,
          help='output file, can have any extension supported by matplotlib')
@args.arg('--input', '-i', required=True, nargs='+',
          help='any number of input files')
@args.arg('--date', '-d', default='job', choices=['build', 'job'],
          help='date to use, either the build/commit date or the date when '
               'the job was run')
@args.arg('--limit', '-l', type=int,
          help='limit the history size to the given number of results')
def history(output, input, date, limit):
    from perftest import plot, result
    results = [result.load(f) for f in input]
    plot.history(results, date, limit).savefig(output)
    log.info(f'Successfully saved plot to {output}')


with log.exception_logging():
    driver()
