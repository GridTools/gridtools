# -*- coding: utf-8 -*-

import sys


if sys.version_info < (3, 6):
    raise Exception('Python 3.6 or newer is required')


class Error(Exception):
    """Base class of all errors inside perftest."""
    pass


class EnvError(Error):
    pass


class NotFoundError(Error):
    pass


class ArgumentError(Error):
    pass


class ParseError(Error):
    pass


class JobError(Error):
    pass


class JobSchedulingError(JobError):
    pass
