"""All eligible training windows, bounded batch memory, and one frozen validation gate."""
import gc
import hashlib
import json
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import mlflow
import mlflow.tensorflow
from mlflow.tracking import MlflowClient
from phm.config import DATA,STATE,FEATURES,SEQ_LEN,SEED,MODEL_NAME
from phm.features import frame,fit_scaler,scale_x,scale_y,inverse_y,metrics
from phm.prepare_fleet import indices
from phm.train import setup_tracking,write_json,passes_gate


class Batches:
    def __init__(self,df,scaler,shuffle=False):
        self.end=indices(df);self.x=scale_x(df[FEATURES].to_numpy(dtype='float32'),scaler)
        self.actual=df.iloc[self.end].ttf_pressure_high.to_numpy()
        self.y=scale_y(self.actual,scaler);self.shuffle=shuffle;self.rng=np.random.default_rng(SEED)
        self.meta=df.iloc[self.end][['Tool','time','time_bin','segment_id','event_id','ttf_pressure_high']].reset_index(drop=True)
    def generator(self):
        order=np.arange(len(self.end))
        if self.shuffle:self.rng.shuffle(order)
        for start in range(0,len(order),256):
            ix=order[start:start+256]
            rows=self.end[ix,None]+np.arange(-SEQ_LEN+1,1)
            yield self.x[rows],self.y[ix]
    def dataset(self):
        d=tf.data.Dataset.from_generator(self.generator,output_signature=(
            tf.TensorSpec((None,SEQ_LEN,len(FEATURES)),tf.float32),tf.TensorSpec((None,),tf.float32)))
        d=d.apply(tf.data.experimental.assert_cardinality((len(self.end)+255)//256))
        opts=tf.data.Options();opts.threading.private_threadpool_size=2
        return d.with_options(opts).prefetch(2)
    def predict(self,model):
        return np.concatenate([model(x,training=False).numpy().ravel() for x,_ in self.generator()])


def train(epochs=15,warm_start=False,training_csv=None):
    setup_tracking();tf.keras.utils.set_random_seed(SEED);started=time.monotonic()
    train_df=pd.read_csv(training_csv) if training_csv else frame('train');val_df=frame('validation')
    pointer_path=STATE/'champion.json';pointer=json.loads(pointer_path.read_text()) if pointer_path.exists() else None
    if training_csv:
        if not train_df['split'].eq('train').all():raise ValueError('Confirmed training split required')
        forbidden=pd.concat([val_df[['Tool','time']],frame('test')[['Tool','time']]])
        if set(zip(train_df.Tool,train_df.time))&set(zip(forbidden.Tool,forbidden.time)):
            raise ValueError('Training overlaps frozen Tool/timestamp holdout')
        if not np.isfinite(train_df[FEATURES+['ttf_pressure_high']]).all().all() or (train_df.ttf_pressure_high<0).any():
            raise ValueError('Finite inputs and confirmed nonnegative targets required')
    ends=indices(train_df);all_y=train_df.iloc[ends].ttf_pressure_high.to_numpy()
    if not len(ends):raise ValueError('No eligible training windows')
    if warm_start:
        if not pointer:raise ValueError('No champion')
        scaler=json.loads((STATE/pointer['bundle']/'scaler.json').read_text())
        model=tf.keras.models.load_model(STATE/pointer['bundle']/'model.keras')
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),loss='huber')
    else:
        scaler=fit_scaler(train_df,all_y);scaler['profile']='fleet';scaler['tool']='fleet'
        scaler['trained_tools']=sorted(train_df.Tool.unique())
        model=tf.keras.Sequential([tf.keras.layers.Input((SEQ_LEN,len(FEATURES))),
            tf.keras.layers.LSTM(32),tf.keras.layers.Dense(16,activation='relu'),tf.keras.layers.Dense(1)])
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),loss='huber')
    tr=Batches(train_df,scaler,True);va=Batches(val_df,scaler)
    print('WINDOWS',len(tr.end),len(va.end),'all eligible train windows, no sampling cap',flush=True)
    baseline_value=float(pointer['baseline_ttf']) if warm_start else float(np.median(all_y))
    baseline=metrics(va.actual,np.full(len(va.actual),baseline_value))
    del train_df,val_df;gc.collect()
    with mlflow.start_run(run_name='fleet-warm-start' if warm_start else 'fleet-all-windows') as run:
        h=model.fit(tr.dataset(),validation_data=va.dataset(),epochs=epochs,verbose=2,
                    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=3,restore_best_weights=True)])
        score=metrics(va.actual,inverse_y(va.predict(model),scaler));comparison=baseline
        if pointer:
            incumbent=tf.keras.models.load_model(STATE/pointer['bundle']/'model.keras')
            inc_scaler=json.loads((STATE/pointer['bundle']/'scaler.json').read_text())
            # Frozen scaler is mandatory in warm start; scratch must evaluate incumbent with its own scaler.
            iv=Batches(frame('validation'),inc_scaler)
            comparison=metrics(iv.actual,inverse_y(iv.predict(incumbent),inc_scaler));del iv,incumbent
        gate=passes_gate(score,baseline,comparison);bundle_name=f'bundles/{run.info.run_id}'
        bundle=STATE/bundle_name;bundle.mkdir(parents=True,exist_ok=True)
        model.save(bundle/'model.keras');write_json(bundle/'scaler.json',scaler)
        per_tool={}
        pred=inverse_y(va.predict(model),scaler)
        for tool,idx in va.meta.groupby('Tool').groups.items():per_tool[tool]=metrics(va.actual[idx],pred[idx])
        report={'run_id':run.info.run_id,'bundle':bundle_name,'validation':score,'baseline_validation':baseline,
                'incumbent_validation':comparison,'gate_passed':gate,'baseline_ttf':baseline_value,
                'gate_policy':'Frozen validation MAE/RMSE/log-MAE <= train-median baseline and incumbent',
                'validation_per_tool':per_tool,'mode':'warm_start' if warm_start else 'scratch',
                'train_sequences':len(tr.end),'validation_sequences':len(va.end),
                'trained_tools':scaler['trained_tools'],'epochs_completed':len(h.history['loss']),
                'seconds':time.monotonic()-started,'profile':'fleet','time_unit':'source_time_unit',
                'sampling':'all eligible windows, stride=1, no 15000 limit','history':h.history,'research_demo_only':True}
        source=__import__('pathlib').Path(training_csv) if training_csv else DATA/'train.parquet'
        digest=hashlib.sha256()
        with source.open('rb') as f:
            for chunk in iter(lambda:f.read(8*1024*1024),b''):digest.update(chunk)
        mlflow.log_params({'epochs':epochs,'seed':SEED,'seq_len':SEQ_LEN,'train_stride':1,
                           'train_windows':len(tr.end),'features':','.join(FEATURES),'profile':'fleet',
                           'training_data_sha256':digest.hexdigest(),'mode':report['mode']})
        for prefix,values in [('validation',score),('baseline_validation',baseline)]:
            for k,v in values.items():mlflow.log_metric(prefix+'_'+k,v)
        mlflow.log_metric('gate_passed',int(gate));mlflow.log_artifact(str(DATA/'dataset_info.json'))
        mlflow.log_artifact(str(bundle/'scaler.json'))
        example=tr.x[tr.end[0]+np.arange(-SEQ_LEN+1,1)][None]
        logged=mlflow.tensorflow.log_model(model,name='model',input_example=example)
        version=mlflow.register_model(logged.model_uri,MODEL_NAME);report['version']=str(version.version)
        client=MlflowClient();client.set_model_version_tag(MODEL_NAME,version.version,'gate_passed',str(gate).lower())
        client.set_model_version_tag(MODEL_NAME,version.version,'bundle',bundle_name)
        if gate:
            client.set_registered_model_alias(MODEL_NAME,'champion',version.version)
            if pointer:write_json(STATE/'previous_champion.json',pointer)
            write_json(pointer_path,report)
        write_json(bundle/'report.json',report);write_json(STATE/'latest_training.json',report)
        mlflow.log_dict(report,'training_report.json')
        print('[GATE PASSED]' if gate else '[GATE FAILED]',json.dumps({k:v for k,v in report.items() if k!='history'}),flush=True)
        return report
