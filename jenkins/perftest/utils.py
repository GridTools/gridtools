# -*- coding: utf-8 -*-

from perftest import ArgumentError
from datetime import datetime, timezone


class Registry(type):
    def __init__(cls, name, bases, attrs):
        if not hasattr(cls, 'registry'):
            cls.registry = []
        else:
            cls.registry.append(cls)

    def instantiate_all(cls, *args, **kwargs):
        return [t(*args, **kwargs) for t in cls.registry]


def timestr(time=None):
    if time is None:
        time = datetime.now(timezone.utc)
    return time.strftime('%Y-%m-%dT%H:%M:%S.%f%z')


def timestr_from_posix(posixtime):
    return timestr(datetime.fromtimestamp(int(posixtime), timezone.utc))


def datetime_from_timestr(tstr):
    try:
        return datetime.strptime(tstr, '%Y-%m-%dT%H:%M:%S.%f%z')
    except ValueError:
        raise ArgumentError(f'"{tstr}" is an invalid time string') from None


def short_timestr(tstr):
    return datetime_from_timestr(tstr).strftime('%y-%m-%d %H:%M')
