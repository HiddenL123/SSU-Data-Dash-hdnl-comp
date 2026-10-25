import sys
from pathlib import Path
import joblib

root = Path.cwd()
sys.path[:0] = [str(root), str(root / 'tests')]
from test_starter import make_listings
from models.features import make_features

train = make_listings(100)
homes = make_listings(12, seed=7)
records = []
for backend in ('catboost', 'lightgbm', 'xgboost'):
    for track in ('off_market', 'list_aware'):
        for name in ('Xpt', 'Xpt_spatiotemporal', 'Xpt_spatiotemporal__selected_binary',
                     'Xpt_spatiotemporal__selected_binary__st_spacetime_splines+geo_context'):
            provider = make_features(name, model=backend, track=track)
            fit = provider.fit_transform(train)
            transformed = provider.transform(homes)
            records.append((name, backend, track, provider, fit, transformed))
dest = root / 'tests/fixtures/repository_organization'
dest.mkdir(parents=True, exist_ok=True)
joblib.dump((train, homes, records), dest / 'before_migration.joblib', compress=3)
print('Captured', len(records), 'pre-migration fitted providers and exact matrices')
