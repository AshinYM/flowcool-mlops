"""Replay different real training windows; never invent sensor values or labels."""
import threading
import numpy as np
from phm.config import FEATURES, SEQ_LEN
from phm.features import frame, windows, scale_x

_lock = threading.Lock()
_pool = None
_cursor = 0


def next_window(scaler):
    global _pool, _cursor
    with _lock:
        if _pool is None:
            df = frame('train')
            x, meta = windows(df)
            # Broad, evenly spaced coverage. Limit to stable inputs for normal replay.
            indices = np.linspace(0, len(meta)-1, min(1024, len(meta)), dtype=int)
            stable = np.abs(scale_x(x[indices], scaler).mean(axis=1)).max(axis=1) <= 5
            candidates = meta.iloc[indices[stable]].copy()
            if len(candidates) < 2:
                raise ValueError('Not enough distinct stable historical windows')
            # Stable shuffle gives reproducible coverage without repeating the last sample.
            candidates = candidates.iloc[np.random.default_rng(42).permutation(len(candidates))]
            _pool = df, candidates.reset_index(drop=True)
        df, candidates = _pool
        item = candidates.iloc[_cursor % len(candidates)]
        _cursor += 1
        rows = df[(df.Tool == item.Tool) & (df.segment_id == item.segment_id)
                  & (df.time_bin <= item.time_bin)].tail(SEQ_LEN)
        columns = ['time', 'time_bin', 'segment_id', 'Tool'] + FEATURES
        import json
        return {'request': {'sequence': json.loads(rows[columns].to_json(orient='records'))},
                'selection': {'source': 'historical_train', 'synthetic': False,
                              'window_id': f'{item.Tool}:{int(item.time)}',
                              'as_of': int(item.time), 'pool_windows': len(candidates)}}
