import numpy as np
import pandas as pd
from phm.config import DATA, FEATURES, SEQ_LEN, PROFILE


def frame(split):
    if PROFILE=='fleet':return pd.read_parquet(DATA/f'{split}.parquet')
    return pd.read_csv(DATA / f'08_M01_pressure_high_{split}.csv')


def windows(df, stride=1, max_windows=None):
    """Final input timestamp owns the target; never cross events, gaps or splits."""
    required = FEATURES + ['time', 'time_bin', 'segment_id', 'event_id', 'Tool']
    if not set(required).issubset(df.columns):
        raise ValueError(f'Missing columns: {sorted(set(required) - set(df.columns))}')
    df = df.reset_index(drop=True)
    finite = np.isfinite(df[FEATURES].to_numpy()).all(axis=1)
    boundary = (df.segment_id.ne(df.segment_id.shift())
                | df.event_id.ne(df.event_id.shift())
                | df.Tool.ne(df.Tool.shift()) | df.time_bin.diff().ne(1)
                | pd.Series(finite).ne(pd.Series(finite).shift()))
    if 'split' in df: boundary |= df.split.ne(df.split.shift())
    arrays, end_indices = [], []
    for _, idx in df.groupby(boundary.cumsum()).groups.items():
        idx = np.asarray(idx)
        if len(idx) < SEQ_LEN or not finite[idx].all(): continue
        ends = np.arange(SEQ_LEN - 1, len(idx), stride)
        if max_windows is not None:
            ends=ends[:max_windows-len(end_indices)]
        values = df.loc[idx, FEATURES].to_numpy(dtype='float32')
        arrays.append(np.stack([values[e-SEQ_LEN+1:e+1] for e in ends]))
        end_indices.extend(idx[ends])
        if max_windows is not None and len(end_indices)>=max_windows:break
    if not arrays: raise ValueError('No continuous 20-row sequences available')
    meta = df.iloc[end_indices].reset_index(drop=True)
    return np.concatenate(arrays), meta


def fit_scaler(df, y):
    values = df[FEATURES].to_numpy(dtype=float)
    mu, sigma = values.mean(axis=0), values.std(axis=0)
    sigma[sigma < 1e-8] = 1
    log_y = np.log1p(y)
    return {'features': FEATURES, 'seq_len': SEQ_LEN, 'mean': mu.tolist(),
            'std': sigma.tolist(), 'target_mean': float(log_y.mean()),
            'target_std': float(max(log_y.std(), 1e-8)), 'target_transform': 'standardized_log1p',
            'time_unit': 'source_time_unit', 'tool': '08M01'}


def scale_x(x, scaler):
    return ((x - np.array(scaler['mean'])) / np.array(scaler['std'])).astype('float32')


def scale_y(y, scaler):
    return ((np.log1p(y) - scaler['target_mean']) / scaler['target_std']).astype('float32')


def inverse_y(y, scaler):
    # Nonnegative TTF, bounded exponent for invalid extrapolation stability.
    return np.maximum(0, np.expm1(np.clip(np.asarray(y) * scaler['target_std']
                                        + scaler['target_mean'], 0, 30)))


def metrics(actual, predicted):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    return {'mae': float(np.abs(actual-predicted).mean()),
            'rmse': float(np.sqrt(np.mean((actual-predicted)**2))),
            'log_mae': float(np.abs(np.log1p(actual)-np.log1p(predicted)).mean())}
