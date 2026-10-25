"""Compatibility alias for :mod:`models.save_model`."""
import sys
from importlib import import_module

_implementation = import_module("models.save_model")

def __getattr__(name):
    return getattr(_implementation, name)

def __dir__():
    return dir(_implementation)
sys.modules[__name__] = _implementation
