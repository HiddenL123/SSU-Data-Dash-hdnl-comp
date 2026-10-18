"""Compatibility import; implementation is in models.legacy.gb_features."""
from models.legacy.gb_features import *  # noqa: F401,F403
from models.legacy import gb_features as _legacy


def __getattr__(name):
    return getattr(_legacy, name)
