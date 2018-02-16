# -*- coding: utf-8 -*-

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
    return datetime.strptime(tstr, '%Y-%m-%dT%H:%M:%S.%f%z')
