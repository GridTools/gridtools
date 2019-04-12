# -*- coding: utf-8 -*-

import contextlib
import logging
import os
import sys
import textwrap


_logger = logging.getLogger('pyutils')
_logger.setLevel(logging.DEBUG)

_formatter = logging.Formatter('%(levelname)s %(asctime)s: %(message)s',
                               '%Y-%m-%d %H:%M:%S')

logfile = os.path.abspath('pyutils.log')

_filehandler = logging.FileHandler(logfile)
_filehandler.setFormatter(_formatter)
_filehandler.setLevel(logging.DEBUG)
_logger.addHandler(_filehandler)

_streamhandler = logging.StreamHandler()
_streamhandler.setFormatter(_formatter)
_streamhandler.setLevel(logging.WARNING)
_logger.addHandler(_streamhandler)

def set_verbosity(level):
    if level <= 0:
        _streamhandler.setLevel(logging.WARNING)
    elif level == 1:
        _streamhandler.setLevel(logging.INFO)
    else:
        _streamhandler.setLevel(logging.DEBUG)


@contextlib.contextmanager
def exception_logging():
    try:
        yield
    except Exception:
        _logger.exception(f'Fatal error: exception was raised, '
                          f'logfile saved at "{logfile}"')
        sys.exit(1)


def _format_message(message, details):
    message = str(message)
    if details is None:
        return message
    details = str(details)
    if details.count('\n') == 0:
        if details.strip() == '':
            details = '[EMPTY]'
        return message + ': ' + details
    else:
        return message + ':\n' + textwrap.indent(details, '    ')


def debug(message, details=None):
    _logger.debug(_format_message(message, details))


def info(message, details=None):
    _logger.info(_format_message(message, details))


def warning(message, details=None):
    _logger.warning(_format_message(message, details))


def error(message, details=None):
    _logger.error(_format_message(message, details))


info(f'Logging to "{logfile}" started')