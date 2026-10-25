"""Compatibility alias for :mod:`feature.comparable_benchmark`."""
import sys
from importlib import import_module

_implementation = import_module("feature.comparable_benchmark")

def __getattr__(name):
    return getattr(_implementation, name)

def __dir__():
    return dir(_implementation)

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
