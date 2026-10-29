"""Create compact, causal PHM2018 CSVs without changing source files.

Run with pandas/numpy available. Defaults resolve relative to this script.
Time units are deliberately left as the source's anonymous time units.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[3]
SOURCE = BASE / 'phm_data_challenge_2018'
DEST = Path(__file__).resolve().parent
TOOL = '08_M01'
BIN_WIDTH = 60
FAULT = 'Flowcool Pressure Too High Check Flowcool Pump'
TARGET = 'TTF_' + FAULT
FEATURES = [
    'FLOWCOOLPRESSURE', 'FLOWCOOLFLOWRATE', 'IONGAUGEPRESSURE',
    'ETCHBEAMVOLTAGE', 'ETCHBEAMCURRENT', 'ETCHSUPPRESSORCURRENT',
    'ETCHGASCHANNEL1READBACK', 'ETCHPBNGASREADBACK',
]
CONTEXT = ['stage', 'recipe', 'recipe_step', 'FIXTURESHUTTERPOSITION']


def main():
    sensor_path = SOURCE / 'train' / f'{TOOL}_DC_train.csv'
    label_path = SOURCE / 'train/train_ttf' / f'{TOOL}_DC_train.csv'
    fault_path = SOURCE / 'train/train_faults' / f'{TOOL}_train_fault_data.csv'
    # Only necessary columns are parsed. Never load all 20 machines.
    frames = list(pd.read_csv(sensor_path, usecols=['time', 'Tool'] + CONTEXT + FEATURES,
                              chunksize=200_000))
    sensor = pd.concat(frames, ignore_index=True)
    del frames
    label = pd.read_csv(label_path, usecols=['time', TARGET])
    assert np.array_equal(sensor.time.to_numpy(), label.time.to_numpy())
    original_rows = len(sensor)
    sensor[TARGET] = label[TARGET].to_numpy()
    del label
    faults = pd.read_csv(fault_path).drop_duplicates()
    events = np.sort(faults.loc[faults.fault_name == FAULT, 'time'].unique())
    all_fault_times = np.sort(faults.time.unique())
    # Repeated timestamps: last recorded state, no future interpolation.
    sensor = sensor.drop_duplicates('time', keep='last').reset_index(drop=True)
    # Hard boundaries across gaps, operating stage/recipe switches, and logged stops.
    stop_epoch = np.searchsorted(all_fault_times, sensor.time, side='right')
    boundary = (sensor.time.diff().gt(BIN_WIDTH)
                | sensor.stage.ne(sensor.stage.shift())
                | sensor.recipe.ne(sensor.recipe.shift())
                | pd.Series(stop_epoch).ne(pd.Series(stop_epoch).shift()))
    sensor['source_segment'] = boundary.cumsum().astype('int64')
    sensor['time_bin'] = sensor.time // BIN_WIDTH
    group = sensor.groupby(['source_segment', 'time_bin'], sort=True)
    agg = group[FEATURES].mean()
    agg['FLOWCOOLPRESSURE_MAX'] = group.FLOWCOOLPRESSURE.max()
    agg['time_start'] = group.time.min()
    agg['time'] = group.time.max()
    agg['sample_count'] = group.size()
    # Use the exact final row, including missing values (group.last skips NaNs).
    endpoint = group.tail(1).set_index(['source_segment', 'time_bin'])
    for c in ['Tool'] + CONTEXT:
        agg[c] = endpoint[c]
    agg['ttf_pressure_high'] = endpoint[TARGET]
    agg = agg.reset_index()
    # After aggregation, no window may cross a missing bin or a source boundary.
    boundary = (agg.source_segment.ne(agg.source_segment.shift())
                | agg.time_bin.diff().ne(1))
    agg['segment_id'] = boundary.cumsum().astype('int64')
    next_idx = np.searchsorted(events, agg.time.to_numpy(), side='left')
    valid_next = next_idx < len(events)
    actual_ttf = agg.ttf_pressure_high.to_numpy()
    labeled = np.isfinite(actual_ttf)
    assert np.all(actual_ttf[labeled] >= 0)
    assert np.all(valid_next[labeled])
    assert np.allclose(actual_ttf[labeled], events[next_idx[labeled]] - agg.time.to_numpy()[labeled])
    agg['event_id'] = np.where(valid_next, next_idx, -1)
    # Group all samples associated with a future F2 event in ONE temporal split.
    n_events = len(events)
    train_end = int(n_events * 0.70)
    val_end = int(n_events * 0.85)
    agg['split'] = np.select(
        [~labeled, next_idx < train_end, next_idx < val_end],
        ['unlabeled', 'train', 'validation'], default='test')
    # Mark eligibility rather than fabricate missing input/target values.
    input_features = FEATURES + ['FLOWCOOLPRESSURE_MAX']
    agg['usable_for_training'] = agg[input_features].notna().all(axis=1) & labeled
    cols = ['time', 'time_start', 'time_bin', 'Tool', 'segment_id', 'event_id', 'split',
            'sample_count', 'usable_for_training'] + CONTEXT + input_features + ['ttf_pressure_high']
    agg = agg[cols]
    DEST.mkdir(parents=True, exist_ok=True)
    agg.to_csv(DEST / '08_M01_pressure_high.csv', index=False, float_format='%.9g')
    for split in ['train', 'validation', 'test']:
        part = agg[agg.split.eq(split) & agg.usable_for_training]
        part.to_csv(DEST / f'08_M01_pressure_high_{split}.csv', index=False, float_format='%.9g')
    faults.to_csv(DEST / '08_M01_fault_events.csv', index=False)
    report = {
        'source_rows': original_rows, 'unique_sensor_timestamps': len(sensor),
        'output_rows': len(agg), 'bin_width_source_time_units': BIN_WIDTH,
        'source_time_unit': 'anonymous; do not assume seconds/minutes',
        'features': input_features, 'context_columns': CONTEXT,
        'target': 'ttf_pressure_high', 'pressure_high_unique_events': n_events,
        'split_event_counts': {'train': train_end, 'validation': val_end-train_end,
                               'test': n_events-val_end},
        'split_rows': agg.groupby('split').size().to_dict(),
        'usable_split_rows': agg[agg.usable_for_training].groupby('split').size().to_dict(),
        'segments': int(agg.segment_id.nunique()),
        'missing_target_rows': int(agg.ttf_pressure_high.isna().sum()),
        'windows_of_20_by_split': {},
    }
    for split in ['train', 'validation', 'test']:
        eligible = agg.split.eq(split) & agg.usable_for_training
        starts = (agg.segment_id.ne(agg.segment_id.shift())
                  | agg.event_id.ne(agg.event_id.shift())
                  | agg.split.ne(agg.split.shift())
                  | eligible.ne(eligible.shift()))
        groups = starts.cumsum()
        lengths = agg[eligible].groupby(groups[eligible]).size()
        report['windows_of_20_by_split'][split] = int((lengths - 19).clip(lower=0).sum())
    # Reopen delivered outputs to validate schema, counts, chronology, event isolation.
    delivered = pd.read_csv(DEST / '08_M01_pressure_high.csv')
    assert len(delivered) == len(agg) and list(delivered.columns) == cols
    assert delivered.time.is_monotonic_increasing and not delivered.time.duplicated().any()
    ids = {}
    for split in ['train', 'validation', 'test']:
        d = pd.read_csv(DEST / f'08_M01_pressure_high_{split}.csv')
        assert d[input_features + ['ttf_pressure_high']].notna().all().all()
        assert d['split'].eq(split).all()
        ids[split] = set(d.event_id)
    assert not (ids['train'] & ids['validation'] or ids['train'] & ids['test']
                or ids['validation'] & ids['test'])
    (DEST / 'dataset_info.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
