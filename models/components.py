"""Library-specific fitting details live here, outside the model container."""

from importlib import import_module

from .defaults import DEFAULT_PARAMS


class EstimatorAdapter:
    def __init__(self, estimator, backend):
        self.estimator = estimator
        self.feature_backend = backend

    def fit(self, X, y):
        if self.feature_backend == "catboost":
            categories = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
            self.estimator.fit(X, y, cat_features=categories)
        else:
            self.estimator.fit(X, y)
        return self

    def predict(self, X):
        return self.estimator.predict(X)


class BoostingComponent(EstimatorAdapter):
    """Adapters use the repository's existing defaults, overridden by kwargs."""

    backend: str
    estimator_name: str

    def __init__(self, **params):
        estimator_class = getattr(import_module(self.backend), self.estimator_name)
        super().__init__(
            estimator_class(**{**DEFAULT_PARAMS[self.backend], **params}), self.backend
        )


def adapt_component(model):
    """Also accept native library estimators in notebook specifications."""
    if isinstance(model, EstimatorAdapter):
        return model
    for cls in type(model).__mro__:
        backend = cls.__module__.split(".")[0]
        if backend in DEFAULT_PARAMS:
            return EstimatorAdapter(model, backend)
    return model
