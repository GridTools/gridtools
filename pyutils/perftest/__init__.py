# -*- coding: utf-8 -*-

import logging
import sys


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
