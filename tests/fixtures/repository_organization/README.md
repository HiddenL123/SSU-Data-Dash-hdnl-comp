# Pre-migration fixtures

Captured from commit `14527cb` before moving any source files. All listings are
synthetic, generated with `tests/test_starter.py::make_listings`; no training
data or production fitted state is embedded here.

- `before_migration.joblib`: 24 fitted feature providers plus their original
  training and prediction matrices (three backends, two tracks, four feature
  configurations). Python 3.13.11, numpy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2,
  and the repository's frozen uv.lock were used.
- `cli_help.json`: exact original help output for all 12 argparse root CLIs.
- `notebook_state.json`: SHA-256 hashes of each notebook excluding cell source;
  protects cell order/count, outputs, execution counts, IDs and metadata.

To regenerate, use a separate checkout of `14527cb`, copy these three generator
scripts into it, and run them from that checkout's root using the locked
environment. Do not regenerate from the refactored branch: that would erase
the independent compatibility baseline.
