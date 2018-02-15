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


def get_timestamp(time=None):
    if time is None:
        time = datetime.now(timezone.utc)
    return time.strftime('%Y-%m-%dT%H:%M:%S.%f%z')


def get_datetime(timestamp):
    return datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%S.%f%z')

