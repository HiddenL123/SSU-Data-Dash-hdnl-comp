"""Compatibility alias for :mod:`models.ensemble_feature_bagging_pilot`."""
import sys
from importlib import import_module

_implementation = import_module("models.ensemble_feature_bagging_pilot")

def __getattr__(name):
    return getattr(_implementation, name)

def __dir__():
    return dir(_implementation)
sys.modules[__name__] = _implementation
