"""Compatibility import; implementation is in models.legacy.basic_models."""
from models.legacy.basic_models import *  # noqa: F401,F403
from models.legacy import basic_models as _legacy


def __getattr__(name):
    return getattr(_legacy, name)
