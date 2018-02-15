# -*- coding: utf-8 -*-

class Error(Exception):
    pass


class ConfigError(Error):
    pass


class NotFoundError(Error):
    pass


class ArgumentError(Error):
    pass


class ParseError(Error):
    pass
