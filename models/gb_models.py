"""Boosting entry points use modular models; explicit legacy imports remain available."""
from models.legacy.gb_models import *  # noqa: F401,F403
from models.legacy import gb_models as _legacy
from models.boosting import GBOffMarketModel, GBListAwareModel, make_model


def __getattr__(name):
    return getattr(_legacy, name)
