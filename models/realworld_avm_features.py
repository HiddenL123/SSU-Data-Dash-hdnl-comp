"""Compatibility alias for :mod:`features.realworld_avm_features`."""
import sys
from importlib import import_module

_implementation = import_module("features.realworld_avm_features")

def __getattr__(name):
    return getattr(_implementation, name)

def __dir__():
    return dir(_implementation)
sys.modules[__name__] = _implementation
