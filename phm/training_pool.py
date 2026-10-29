"""Rotate confirmed historical train subsets for automatic retraining."""
import hashlib
import json
import os
from pathlib import Path
import pandas as pd
from phm.config import DATA, STATE

# Prespecified replay sequence; outcomes are decided by the unchanged gate.
# Different cohorts/ranges model changing available training batches.
SPECS = [
    ('event-balanced-a', '정비 기록 A', 86400, 11, None),
    ('full-history', '전체 운전 기록', None, None, None),
    ('event-balanced-b', '정비 기록 B', 86400, 12, None),
    ('near-43200', '근접 구간 A', 43200, None, None),
    ('event-cohort-d', '사건 묶음 D', 120000, None, 3),
    ('near-86400', '근접 구간 B', 86400, None, None),
    ('near-70000', '근접 구간 C', 70000, None, None),
    ('omit-0', '정비 기록 C', 86400, 0, None),
    ('cohort-b', '사건 묶음 B', 120000, None, 1),
    ('near-57600', '근접 구간 D', 57600, None, None),
    ('omit-9', '정비 기록 D', 86400, 9, None),
    ('cohort-c', '사건 묶음 C', 120000, None, 2),
    ('near-120000', '근접 구간 E', 120000, None, None),
    ('omit-8', '정비 기록 E', 86400, 8, None),
    ('near-240000', '운전 구간 A', 240000, None, None),
    ('cohort-e', '사건 묶음 E', 120000, None, 4),
]


def prepare(inspect):
    source = DATA/'08_M01_pressure_high_train.csv'
    df = pd.read_csv(source, float_precision='round_trip')
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    folder = STATE/'retraining_pool'
    folder.mkdir(parents=True, exist_ok=True)
    ids = sorted(df.event_id.unique())
    result = []
    for key, name, cutoff, omitted, cohort in SPECS:
        rows = df if cutoff is None else df[df.ttf_pressure_high <= cutoff]
        if omitted is not None:
            rows = rows[rows.event_id.ne(omitted)]
        if cohort is not None:
            events = [e for i, e in enumerate(ids) if i % 5 == cohort]
            rows = rows[rows.event_id.isin(events)]
        path = source if cutoff is None else folder/f'{key}.csv'
        if path != source:
            # Rebuild from the same source on startup; no sensor/label generation.
            rows.to_csv(path, index=False)
        info = inspect(rows)
        if not info['retrain_eligible']:
            raise ValueError(f'Ineligible automatic training cohort: {key}')
        result.append({'upload_id': 'auto-'+key, 'filename': path.name, 'path': str(path),
                       'name': name, 'source': 'historical_train_cohort',
                       'training_focus': 'full' if cutoff is None else 'automatic_pool',
                       'source_sha256': source_hash,
                       'data_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                       'selection_rule': {'ttf_max': cutoff, 'omitted_event': omitted, 'event_mod5': cohort},
                       **info})
    return result


def cursor():
    path = STATE/'retraining_cursor.json'
    return int(json.loads(path.read_text())['next_index']) if path.exists() else 0


def advance(index):
    path = STATE/'retraining_cursor.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps({'next_index': index+1}))
    os.replace(temporary, path)


def status(pool):
    index = cursor()
    return {'datasets': len(pool), 'next_index': index,
            'next_dataset_name': pool[index % len(pool)]['name'] if pool else None,
            'records': [{k:v for k,v in item.items() if k!='path'} for item in pool]}
