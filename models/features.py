"""Compatibility alias for :mod:`features.features`."""
import sys
from importlib import import_module

_implementation = import_module("features.features")

def __getattr__(name):
    return getattr(_implementation, name)

def __dir__():
    return dir(_implementation)
sys.modules[__name__] = _implementation
