# -*- coding: utf-8 -*-


class Registry(type):
    def __init__(cls, name, bases, attrs):
        if not hasattr(cls, 'registry'):
            cls.registry = []
        else:
            cls.registry.append(cls)

    def instantiate_all(cls, *args, **kwargs):
        return [t(*args, **kwargs) for t in cls.registry]
