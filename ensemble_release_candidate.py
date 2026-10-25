"""Compatibility alias for :mod:`models.ensemble_release_candidate`."""
import sys
from importlib import import_module

_implementation = import_module("models.ensemble_release_candidate")

def __getattr__(name):
    return getattr(_implementation, name)

def __dir__():
    return dir(_implementation)
sys.modules[__name__] = _implementation
