"""Compatibility alias for :mod:`feature.spacetime_full_feature_comparison`."""
import sys
from importlib import import_module

_implementation = import_module("feature.spacetime_full_feature_comparison")

def __getattr__(name):
    return getattr(_implementation, name)

def __dir__():
    return dir(_implementation)
sys.modules[__name__] = _implementation
