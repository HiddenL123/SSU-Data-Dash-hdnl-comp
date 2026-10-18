"""Explicit access to preserved upstream baselines inside the models package.

The original starter tests use these baselines. Production imports models.model.
"""
from .legacy.upstream_models import (
    ListPriceBaseline, NearestSalesBaseline, FeatureModel, make_model,
)
