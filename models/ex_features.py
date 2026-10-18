"""Compatibility import; implementation is in models.legacy.ex_features."""
from models.legacy.ex_features import *  # noqa: F401,F403
from models.legacy import ex_features as _legacy


def __getattr__(name):
    return getattr(_legacy, name)
