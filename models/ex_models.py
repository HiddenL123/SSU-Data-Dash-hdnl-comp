"""Compatibility import; implementation is in models.legacy.ex_models."""
from models.legacy.ex_models import *  # noqa: F401,F403
from models.legacy import ex_models as _legacy


def __getattr__(name):
    return getattr(_legacy, name)
