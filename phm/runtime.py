import json
import os
import threading
import time
from collections import deque
from datetime import datetime, timezone
import numpy as np
from phm.config import STATE, FEATURES, MODEL_NAME, TRACKING_URI
from phm.features import scale_x, inverse_y, metrics

LOCK=threading.RLock()
EVENTS=deque(maxlen=300)
QUALITY=deque(maxlen=100)
CACHE=None
REQUESTS={'count':0,'errors':0,'server_errors':0,'latencies_ms':deque(maxlen=500),
          'recent':deque(maxlen=5000),'started':time.monotonic()}


def event(kind, **details):
    item={'at':datetime.now(timezone.utc).isoformat(),'kind':kind,**details}
    with LOCK:
        EVENTS.append(item)
        with (STATE/'aiops.jsonl').open('a') as f:f.write(json.dumps(item,ensure_ascii=False)+'\n')
    return item


def pointer():
    path=STATE/'champion.json'
    if not path.exists():raise FileNotFoundError('No model passed the validation gate; run python -m phm.train')
    return json.loads(path.read_text())


def loaded():
    global CACHE
    with LOCK:
        p=pointer()
        source=os.getenv('MODEL_SOURCE','local')
        if CACHE is None or CACHE['version']!=p['version'] or CACHE['source']!=source:
            import tensorflow as tf
            scaler=json.loads((STATE/p['bundle']/'scaler.json').read_text())
            if source=='mlflow':
                import mlflow,mlflow.tensorflow
                mlflow.set_tracking_uri(TRACKING_URI)
                model=mlflow.tensorflow.load_model(f'models:/{MODEL_NAME}/{p["version"]}')
            else:model=tf.keras.models.load_model(STATE/p['bundle']/'model.keras')
            if CACHE is not None and CACHE['version']!=p['version']:QUALITY.clear()
            CACHE={'version':p['version'],'source':source,'model':model,'scaler':scaler,'report':p}
            event('model_loaded',version=p['version'],source=source)
        return CACHE


def predict(x):
    cache=loaded()
    scaled=scale_x(np.asarray(x,dtype='float32'),cache['scaler'])
    with LOCK:raw=cache['model'](scaled,training=False).numpy().ravel()
    values=inverse_y(raw,cache['scaler'])
    if not np.isfinite(values).all():raise ValueError('Nonfinite model output')
    return values,cache


def monitor(x, predicted=None, actual=None):
    cache=loaded()
    z=scale_x(np.asarray(x),cache['scaler'])
    shift=np.abs(z.reshape(-1,len(FEATURES)).mean(axis=0))
    names=[FEATURES[i] for i,v in enumerate(shift) if v>5.0]
    quality=None
    if actual is not None:
        pairs=list(zip(np.asarray(actual).ravel(),np.asarray(predicted).ravel()))
        with LOCK:
            QUALITY.extend(pairs);snapshot=list(QUALITY)
        quality={'confirmed_labels':len(snapshot),'metrics':None,'degraded':False}
        if len(snapshot)>=21:
            a,p=np.array(snapshot).T
            quality['metrics']=metrics(a,p)
            quality['degraded']=quality['metrics']['log_mae']>max(cache['report']['validation']['log_mae']*1.5,0.1)
    status='input_shift' if names else 'stable_input'
    if quality and quality['degraded']:status='quality_degraded'
    result={'status':status,'shifted_features':names,'max_mean_abs_z':float(shift.max()),
            'input_shift_threshold':5.0,'quality':quality,
            'model_version':cache['version'],
            'policy':'Input change raises an advisory; retrain only with labeled training data and frozen gate.'}
    if names or (quality and quality['degraded']):event('monitor_alert',**result)
    return result


def request_record(path,method,status,duration):
    item={'ts':time.time(),'path':path,'method':method,'status':status,'duration_ms':duration}
    with LOCK:
        REQUESTS['count']+=1;REQUESTS['errors']+=int(status>=400);REQUESTS['server_errors']+=int(status>=500)
        REQUESTS['latencies_ms'].append(duration);REQUESTS['recent'].append(item)
        with (STATE/'requests.jsonl').open('a') as f:f.write(json.dumps(item)+'\n')


def service_metrics():
    with LOCK:recent=[r for r in REQUESTS['recent'] if time.time()-r['ts']<=300]
    lat=[r['duration_ms'] for r in recent];n=len(recent)
    rate=sum(r['status']>=500 for r in recent)/n if n else 0
    average=float(np.mean(lat)) if lat else None
    return {'window_seconds':300,'request_count':n,'avg_latency_ms':average,
            'p95_ms':float(np.percentile(lat,95)) if lat else None,'error_rate_5xx':rate,
            'validation_errors_4xx':sum(400<=r['status']<500 for r in recent),
            'scope':'POST operations in the last 300s; dashboard GET polling excluded',
            'requests_per_second_observed':n/min(300,max(time.monotonic()-REQUESTS['started'],1)),
            'thresholds':{'avg_latency_ms':1000,'error_rate_5xx':0.05,'minimum_requests':21},
            'alert':bool(n>=21 and ((average or 0)>1000 or rate>0.05))}


def rollback():
    from phm.train import write_json,setup_tracking
    from mlflow.tracking import MlflowClient
    path=STATE/'previous_champion.json'
    if not path.exists():raise ValueError('No previous champion')
    with LOCK:
        old=json.loads(path.read_text());current=pointer()
        setup_tracking();MlflowClient().set_registered_model_alias(MODEL_NAME,'champion',old['version'])
        write_json(STATE/'champion.json',old);write_json(path,current)
        QUALITY.clear()
    event('rollback',from_version=current['version'],to_version=old['version'])
    return old['version']
