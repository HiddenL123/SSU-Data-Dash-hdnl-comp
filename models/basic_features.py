"""Compatibility import; implementation is in models.legacy.basic_features."""
from models.legacy.basic_features import *  # noqa: F401,F403
from models.legacy import basic_features as _legacy


def __getattr__(name):
    return getattr(_legacy, name)
