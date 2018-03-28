# -*- coding: utf-8 -*-

import contextlib
import logging
import sys
import textwrap


if sys.version_info < (3, 6):
    raise Exception('Python 3.6 or newer is required')


class Error(Exception):
    """Base class of all errors inside perftest."""
    pass


class ConfigError(Error):
    pass


class NotFoundError(Error):
    pass


class ArgumentError(Error):
    pass


class ParseError(Error):
    pass


class JobError(Error):
    pass


logger = logging.getLogger(__name__)

loghandler = logging.StreamHandler()
loghandler.setFormatter(
        logging.Formatter('%(levelname)s %(asctime)s: %(message)s',
                          '%Y-%m-%d %H:%M:%S'))

logger.addHandler(loghandler)


def set_verbose(verbosity):
    """Sets the global verbosity of the logger.

    If `verbosity` is 0, only errors and warnings are logged, if `verbosity`
    is 1 additional info messages are logged. If `verbosity` is 2 or higher,
    all debug messages are logged.

    Args:
        verbosity: Integer, higher means more verbose.
    """
    if verbosity == 0:
        logger.setLevel(logging.WARNING)
    elif verbosity == 1:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.DEBUG)


set_verbose(0)


@contextlib.contextmanager
def exception_logging():
    try:
        yield
    except Exception:
        logger.exception('Fatal error: exception was raised')
        sys.exit(1)


def _multiline_redirect(func):
    def multiline_func(message, details=None, **kwargs):
        if details is None:
            func(message, **kwargs)
        else:
            func(message + '\n' + textwrap.indent(details, '    '), **kwargs)
    return multiline_func


logger.debug = _multiline_redirect(logger.debug)
logger.info = _multiline_redirect(logger.info)
logger.warning = _multiline_redirect(logger.warning)
logger.error = _multiline_redirect(logger.error)
