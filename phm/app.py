import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','2')
os.environ.setdefault('MLFLOW_DISABLE_AGENT_HINT','1')
os.environ.setdefault('MPLCONFIGDIR','/tmp/phm-matplotlib')
import io
import json
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Query
from fastapi.responses import FileResponse,Response
from pydantic import BaseModel, ConfigDict, Field, model_validator, create_model
from phm.config import STATE, DATA, FEATURES, SEQ_LEN, ROOT, PROFILE, TOOLS
from phm.features import windows, frame
from phm import runtime

JOBS={}
UPLOADS={}
AUTO_POOL=[]
JOB_LOCK=__import__('threading').Lock()
PREPARED_TRAIN_ID='prepared-08M01-train'
FOCUS_TRAIN_ID='prepared-08M01-maintenance-focus'


class PointBase(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    time: int=Field(ge=0)
    time_bin: int=Field(ge=0)
    segment_id: int=Field(ge=0)
    Tool: str=Field(pattern='^(?:'+'|'.join(TOOLS)+')$')

Point=create_model('Point',__base__=PointBase,**{name:(float,...) for name in FEATURES})


class Prediction(BaseModel):
    model_config=ConfigDict(extra='forbid')
    sequence:list[Point]=Field(min_length=SEQ_LEN,max_length=SEQ_LEN)

    @model_validator(mode='after')
    def continuous(self):
        p=self.sequence
        for a,b in zip(p,p[1:]):
            if b.time<=a.time or b.time_bin!=a.time_bin+1 or b.segment_id!=a.segment_id or b.Tool!=a.Tool:
                raise ValueError('Sequence must be chronological, consecutive bins in one segment')
        return self


class Batch(BaseModel):
    sequences:list[Prediction]=Field(min_length=1,max_length=128)
    actual_ttf:list[float]|None=None
    auto_retrain_upload_id:str|None=None
    auto_response:bool=True
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)

    @model_validator(mode='after')
    def labels(self):
        if self.actual_ttf is not None:
            if len(self.actual_ttf)!=len(self.sequences) or any(x<0 for x in self.actual_ttf):
                raise ValueError('Confirmed nonnegative labels must match prediction count')
        return self


class Retrain(BaseModel):
    model_config=ConfigDict(extra='forbid')
    upload_id:str
    epochs:int=Field(default=4,ge=1,le=20)


@asynccontextmanager
async def lifespan(app):
    connect_prepared_training()
    if os.getenv('LOADING_MODE','lazy')=='eager':runtime.loaded()
    yield


app=FastAPI(title='Wafer Flowcool · LSTM & AIOps',lifespan=lifespan)


@app.middleware('http')
async def latency(request,call_next):
    start=time.perf_counter();status=500
    try:
        response=await call_next(request)
        status=response.status_code
        return response
    finally:
        # Dashboard polling must not inflate inference traffic or hide error rates.
        if request.method == 'POST':
            runtime.request_record(request.url.path,request.method,status,(time.perf_counter()-start)*1000)


@app.get('/')
def dashboard():return FileResponse(ROOT/'serving_app/static'/('phm_fleet.html' if PROFILE=='fleet' else 'phm.html'))


@app.get('/favicon.ico',include_in_schema=False)
def favicon():return Response(status_code=204)


@app.get('/health')
def health():
    try:p=runtime.pointer()
    except FileNotFoundError:p={}
    current=runtime.CACHE is not None and runtime.CACHE['version']==p.get('version')
    return {'status':'ready' if current else ('model_available' if p else 'no_validated_model'),'model_loaded':current,
            'model_version':p.get('version'),'model_source':os.getenv('MODEL_SOURCE','local'),
            'loading_mode':os.getenv('LOADING_MODE','lazy'),'tool':'fleet' if PROFILE=='fleet' else '08M01',
            'profile':PROFILE,'supported_tools':TOOLS,'time_unit':'source_time_unit'}


def input_array(req):return [[getattr(p,c) for c in FEATURES] for p in req.sequence]


class PredictionResponse(BaseModel):
    model_config=ConfigDict(allow_inf_nan=False)
    tool:str
    as_of:int
    predicted_ttf:float=Field(ge=0)
    time_unit:str
    model_version:str
    usage:str
    reliability_warning:str
    monitoring:dict


def reliability():
    if PROFILE!='fleet':return 'Whole-horizon holdout TTF underperformed the constant baseline; not validated for production decisions.'
    p=STATE/'evaluation.json'
    if not p.exists():return 'Research demo; independent equipment holdout has not yet been evaluated.'
    e=json.loads(p.read_text())
    return ('Independent equipment holdout beats the train-median baseline in MAE/RMSE/log-MAE; industrial benefit is unproven.'
            if e.get('beats_baseline') else 'Independent equipment holdout does not beat all baseline metrics; industrial benefit is unproven.')


@app.post('/predict',response_model=PredictionResponse)
def predict(req:Prediction):
    try:values,cache=runtime.predict([input_array(req)])
    except FileNotFoundError as e:raise HTTPException(503,str(e))
    monitoring=runtime.monitor([input_array(req)])
    end=req.sequence[-1]
    return {'tool':end.Tool,'as_of':end.time,'predicted_ttf':float(values[0]),
            'time_unit':'source_time_unit','model_version':cache['version'],
            'usage':'maintenance decision support',
            'reliability_warning':reliability(),
            'monitoring':monitoring}


@app.post('/predict/batch-test')
def batch(req:Batch,tasks:BackgroundTasks):
    try:
        x=np.array([input_array(r) for r in req.sequences])
        values,cache=runtime.predict(x)
    except FileNotFoundError as e:raise HTTPException(503,str(e))
    check=runtime.monitor(x,values,req.actual_ttf)
    result={'predictions':values.tolist(),'model_version':cache['version'],
            'drift_check':check,'time_unit':'source_time_unit'}
    if check['status'] in ['input_shift','quality_degraded'] and req.auto_response:
        result['response_job']=queue_job(Retrain(upload_id=req.auto_retrain_upload_id or 'automatic'),tasks)
    return result


@app.get('/data/example')
def example(tool:str=Query(default='08M01',pattern='^(?:'+'|'.join(TOOLS)+')$')):
    if PROFILE=='fleet':
        path=DATA/'examples'/f'{tool}_first.json'
        if not path.exists():raise HTTPException(404,'No continuous historical sequence for this Tool')
        return json.loads(path.read_text())
    df=pd.read_csv(DATA/'08_M01_pressure_high_train.csv')
    _,m=windows(df,max_windows=1)
    # Find the first valid 20-row window, not simply the first 20 CSV rows.
    last=m.iloc[0]
    rows=df[(df.segment_id==last.segment_id)&(df.time_bin<=last.time_bin)].tail(SEQ_LEN)
    columns=['time','time_bin','segment_id','Tool']+FEATURES
    return {'sequence':json.loads(rows[columns].to_json(orient='records'))}


@app.get('/data/status')
def data_status():
    from phm.training_pool import status
    return {'retraining_pool':status(AUTO_POOL) if PROFILE=='single' else None,
            'prepared':json.loads((DATA/'dataset_info.json').read_text()),
            'default_training':{k:v for k,v in UPLOADS.get(PREPARED_TRAIN_ID,{}).items() if k!='path'} or None,
            'focus_training':{k:v for k,v in UPLOADS.get(FOCUS_TRAIN_ID,{}).items() if k!='path'} or None,
            'uploads':[{k:v for k,v in u.items() if k!='path'} for u in UPLOADS.values()]}


@app.get('/data/replay')
def replay():
    if PROFILE!='single':raise HTTPException(400,'Historical replay supports the single-tool profile')
    from phm.replay import next_window
    return next_window(runtime.loaded()['scaler'])


def inspect_dataset(df):
    x,_=windows(df)
    if any(not g.time.is_monotonic_increasing or g.time.duplicated().any() for _,g in df.groupby('Tool')):
        raise ValueError('CSV time must increase uniquely within each Tool')
    if not df.Tool.isin(TOOLS).all():raise ValueError('Unsupported Tool')
    if not np.isfinite(df[FEATURES].to_numpy()).all():raise ValueError('Missing/nonfinite sensor values')
    labeled='ttf_pressure_high' in df and np.isfinite(df.ttf_pressure_high).all() and (df.ttf_pressure_high>=0).all()
    train_only='split' in df and df['split'].eq('train').all()
    if PROFILE=='fleet':
        holdout=pd.concat([pd.read_parquet(DATA/f'{s}.parquet',columns=['Tool','time']) for s in ['validation','test']])
        overlaps=bool(set(zip(df.Tool,df.time))&set(zip(holdout.Tool,holdout.time)))
    else:
        holdout=pd.concat([pd.read_csv(DATA/f'08_M01_pressure_high_{s}.csv',usecols=['time']) for s in ['validation','test']])
        overlaps=bool(set(df.time)&set(holdout.time))
    return {'rows':len(df),'sequences':len(x),'confirmed_labels':bool(labeled),
            'retrain_eligible':bool(labeled and train_only and not overlaps and len(x)>=128),
            'holdout_overlap':overlaps}


def connect_prepared_training():
    if PROFILE!='single':return
    for dataset_id,filename,focus in [
        (FOCUS_TRAIN_ID,'08_M01_maintenance_focus_train.csv','maintenance_focus'),
        (PREPARED_TRAIN_ID,'08_M01_pressure_high_train.csv','full')]:
        path=DATA/filename
        info={'upload_id':dataset_id,'path':str(path),'filename':path.name,
              'source':'prepared_train','training_focus':focus,**inspect_dataset(pd.read_csv(path))}
        if not info['retrain_eligible']:raise ValueError('Prepared training data failed validation')
        UPLOADS[dataset_id]=info
        runtime.event('dataset_connected',upload_id=dataset_id,rows=info['rows'],
                      sequences=info['sequences'],retrain_eligible=True,training_focus=focus)

    from phm.training_pool import prepare
    AUTO_POOL[:] = prepare(inspect_dataset)
    UPLOADS.update({item['upload_id']:item for item in AUTO_POOL})
    runtime.event('training_pool_ready',datasets=len(AUTO_POOL))


@app.post('/data/upload')
async def upload(file:UploadFile=File(...)):
    raw=await file.read(60*1024*1024+1)
    if len(raw)>60*1024*1024:raise HTTPException(413,'Maximum CSV size is 60 MiB')
    try:
        df=pd.read_csv(io.BytesIO(raw))
        metadata=inspect_dataset(df)
    except (ValueError,KeyError,TypeError,UnicodeDecodeError) as e:raise HTTPException(422,str(e))
    upload_id=uuid.uuid4().hex
    path=STATE/'uploads'/f'{upload_id}.csv';path.parent.mkdir(exist_ok=True);path.write_bytes(raw)
    info={'upload_id':upload_id,'path':str(path),**metadata,
          'source':'uploaded_csv','filename':Path(file.filename or 'data.csv').name,
          'training_focus':'maintenance_focus' if metadata['confirmed_labels'] and df.ttf_pressure_high.max()<=86400 else 'full'}
    UPLOADS[upload_id]=info
    runtime.event('csv_uploaded',upload_id=upload_id,rows=len(df),retrain_eligible=info['retrain_eligible'])
    return {k:v for k,v in info.items() if k!='path'}


def run_job(job_id,path,epochs):
    from phm.train import train
    JOBS[job_id]['status']='running'
    runtime.event('retrain_triggered',job_id=job_id,mode='warm_start',
                  source_upload_id=JOBS[job_id]['upload_id'])
    try:
        report=train(epochs,warm_start=True,training_csv=path,
                     training_focus=UPLOADS[JOBS[job_id]['upload_id']].get('training_focus','full'))
        JOBS[job_id].update(status='completed',result={k:v for k,v in report.items() if k!='history'})
        runtime.event('candidate_promoted' if report['gate_passed'] else 'candidate_rejected',
                      job_id=job_id,version=report['version'],validation=report['validation'])
        if report['gate_passed']:runtime.QUALITY.clear()
    except Exception as e:
        JOBS[job_id].update(status='failed',error=str(e));runtime.event('retrain_failed',job_id=job_id,error=str(e))
    finally:JOB_LOCK.release()


def queue_job(req,tasks):
    automatic=req.upload_id=='automatic'
    if automatic and not AUTO_POOL:raise HTTPException(409,'No automatic training pool available')
    if not automatic:
        info=UPLOADS.get(req.upload_id)
        if not info:raise HTTPException(404,'Upload not found in this server session')
        if not info['retrain_eligible']:raise HTTPException(422,'Requires confirmed train labels, >=128 sequences, and no frozen holdout overlap')
    try:runtime.pointer()
    except FileNotFoundError:raise HTTPException(409,'No champion to fine-tune')
    if not JOB_LOCK.acquire(blocking=False):raise HTTPException(409,'Training already running')
    try:
        if automatic:
            from phm.training_pool import cursor,advance
            index=cursor();info=AUTO_POOL[index % len(AUTO_POOL)];advance(index)
            runtime.event('dataset_selected',upload_id=info['upload_id'],rows=info['rows'],
                          sequences=info['sequences'],dataset=info['name'])
        job_id=uuid.uuid4().hex
        JOBS[job_id]={'job_id':job_id,'status':'queued','upload_id':info['upload_id'],
                      'training_dataset':info.get('name',info['filename']),
                      'training_data_sha256':info.get('data_sha256'), 'automatic_selection':automatic}
        tasks.add_task(run_job,job_id,info['path'],req.epochs)
        return JOBS[job_id].copy()
    except Exception:
        JOB_LOCK.release()
        raise


@app.post('/models/retrain',status_code=202)
def retrain(req:Retrain,tasks:BackgroundTasks):return queue_job(req,tasks)


@app.post('/monitoring/auto-response',status_code=202)
def auto_response(req:Retrain,tasks:BackgroundTasks):
    from datetime import datetime, timezone
    try:current=runtime.pointer()['version']
    except FileNotFoundError:raise HTTPException(409,'No champion to fine-tune')
    now=datetime.now(timezone.utc)
    with runtime.LOCK:
        alerts=[e for e in runtime.EVENTS if e['kind']=='monitor_alert'
                and e.get('model_version')==current
                and 0 <= (now-datetime.fromisoformat(e['at'])).total_seconds() <= 300]
    if not alerts:raise HTTPException(409,'No current-version alert in the last 300 seconds; use explicit retrain for routine refresh')
    return queue_job(req,tasks)


@app.get('/jobs/{job_id}')
def job(job_id:str):
    if job_id not in JOBS:raise HTTPException(404,'Unknown job')
    return JOBS[job_id]


@app.post('/models/rollback')
def rollback():
    if JOB_LOCK.locked():raise HTTPException(409,'Training running')
    try:return {'model_version':runtime.rollback()}
    except ValueError as e:raise HTTPException(409,str(e))


@app.get('/monitoring/status')
def monitoring():
    lat=list(runtime.REQUESTS['latencies_ms'])
    return {'requests':runtime.REQUESTS['count'],'errors':runtime.REQUESTS['errors'],
            'p95_ms':float(np.percentile(lat,95)) if lat else None,
            'confirmed_label_count':len(runtime.QUALITY),'events':list(runtime.EVENTS)[-50:],
            'service_metrics':runtime.service_metrics()}


@app.get('/metrics/summary')
def metrics_summary():return runtime.service_metrics()


@app.get('/operations/summary')
def operations_summary():
    from phm.operations import summary
    return summary(JOBS)


@app.get('/logs')
def logs():
    path=STATE/'aiops.jsonl'
    return {'events':[json.loads(s) for s in path.read_text().splitlines()[-100:]] if path.exists() else []}


@app.get('/models/status')
def models():
    def read(name):
        p=STATE/name
        if not p.exists():return None
        return {k:v for k,v in json.loads(p.read_text()).items() if k!='history'}
    return {'champion':read('champion.json'),'latest_candidate':read('latest_training.json')}


@app.get('/evaluation')
def evaluation():
    path=STATE/'evaluation.json'
    return json.loads(path.read_text()) if path.exists() else {'status':'not_evaluated'}


@app.get('/research/fleet')
def fleet_research():
    """Read-only experiment status; never changes the serving profile or alias."""
    state=ROOT/'phm_state_fleet';data=ROOT/'data/phm2018_fleet'
    def read(path):return json.loads(path.read_text()) if path.exists() else None
    candidate=read(state/'latest_training.json')
    if candidate:candidate.pop('history',None)
    return {'dataset':read(data/'dataset_info.json'),'candidate':candidate,
            'evaluation':read(state/'evaluation.json'),
            'fleet_deployed':(state/'champion.json').exists(),
            'current_serving_profile':PROFILE,
            'purpose':'All-equipment experiment record; a rejected candidate is not served.'}


@app.get('/fleet/status')
def fleet_status():
    if PROFILE!='fleet':raise HTTPException(404,'Available in fleet profile')
    reqs=[]
    for tool in TOOLS:
        path=DATA/'examples'/f'{tool}_latest.json'
        if path.exists():reqs.append(Prediction.model_validate(json.loads(path.read_text())))
    try:values,cache=runtime.predict([input_array(r) for r in reqs])
    except FileNotFoundError as e:raise HTTPException(503,str(e))
    records=[{'tool':r.sequence[-1].Tool,'as_of':r.sequence[-1].time,'predicted_ttf':float(v),
              'training_supported':r.sequence[-1].Tool in cache['report'].get('trained_tools',[])}
             for r,v in zip(reqs,values)]
    records.sort(key=lambda r:r['predicted_ttf'])
    return {'mode':'historical_replay','equipment':records,'model_version':cache['version'],
            'time_unit':'source_time_unit','usage':'Inspection review ordering, not automatic shutdown',
            'reliability_warning':reliability()}
