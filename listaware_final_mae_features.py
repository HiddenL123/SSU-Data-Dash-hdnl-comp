"""Compatibility alias for :mod:`feature.listaware_final_mae_features`."""
import sys
from importlib import import_module

_implementation = import_module("feature.listaware_final_mae_features")

def __getattr__(name):
    return getattr(_implementation, name)

def __dir__():
    return dir(_implementation)

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
