"""Compatibility import; implementation is in models.legacy.dummy_model."""
from models.legacy.dummy_model import *  # noqa: F401,F403
from models.legacy import dummy_model as _legacy


def __getattr__(name):
    return getattr(_legacy, name)
