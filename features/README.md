# Reusable features

Use `from features import make_features` or import a provider from its canonical module.
These implementations depend on listing data and third-party libraries, not model architectures.
`gb_features.py` contains the original boosting encoder implementation; the old
`models.gb_features` and `models.legacy.gb_features` paths both remain importable.
No feature identifiers, fitted state, schema, numerical logic or defaults changed.
