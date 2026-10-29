"""Small LSTM with frozen validation, MLflow registry and nonregression gate."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '2')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '2')
os.environ.setdefault('MLFLOW_DISABLE_AGENT_HINT', '1')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/phm-matplotlib')
import argparse
import hashlib
import json
import time
import uuid
from datetime import datetime, timezone
import numpy as np
import tensorflow as tf
import mlflow
import mlflow.tensorflow
from mlflow.tracking import MlflowClient
from phm.config import DATA, STATE, FEATURES, SEQ_LEN, SEED, MODEL_NAME, TRACKING_URI, PROFILE
from phm.features import frame, windows, fit_scaler, scale_x, scale_y, inverse_y, metrics


def setup_tracking():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment('wafer_flowcool_fleet' if PROFILE=='fleet' else 'wafer_flowcool_pressure_high')


def write_json(path, obj):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    temp.replace(path)


def passes_gate(score,baseline,incumbent):
    return (all(np.isfinite(score[k]) and score[k]<=baseline[k] for k in ['mae','rmse','log_mae'])
            and all(score[k]<=incumbent[k] for k in ['mae','rmse','log_mae']))


class NonregressionCheckpoint(tf.keras.callbacks.Callback):
    """Choose a trained epoch with actual improvement on frozen validation."""
    def __init__(self, inputs, labels, scaler, baseline, incumbent):
        super().__init__()
        self.inputs,self.labels,self.scaler=inputs,labels,scaler
        self.baseline,self.incumbent=baseline,incumbent
        self.best_weights=None
        self.best_score=None
        self.best_epoch=None

    def on_epoch_end(self,epoch,logs=None):
        score=metrics(self.labels,inverse_y(self.model(self.inputs,training=False).numpy().ravel(),self.scaler))
        if (passes_gate(score,self.baseline,self.incumbent)
                and any(score[k]<self.incumbent[k] for k in score)
                and (self.best_score is None or score['rmse']<self.best_score['rmse'])):
            self.best_weights=self.model.get_weights()
            self.best_score=score
            self.best_epoch=epoch+1


def train(epochs=15, warm_start=False, limit=15000, training_csv=None, training_focus='full'):
    if PROFILE=='fleet':
        from phm.fleet_train import train as fleet_train
        return fleet_train(epochs,warm_start,training_csv)
    setup_tracking()
    tf.keras.utils.set_random_seed(SEED)
    import pandas as pd
    train_df = pd.read_csv(training_csv) if training_csv else frame('train')
    val_df = frame('validation')
    if training_csv:
        if 'split' not in train_df or not train_df['split'].eq('train').all():
            raise ValueError('Retraining requires explicitly labeled train rows')
        holdout=pd.concat([val_df,frame('test')])
        if set(train_df.time)&set(holdout.time):raise ValueError('Training upload overlaps frozen holdout timestamps')
        if not np.isfinite(train_df[FEATURES+['ttf_pressure_high']].to_numpy()).all():
            raise ValueError('Retraining requires finite inputs and confirmed TTF labels')
        if (train_df.ttf_pressure_high<0).any():raise ValueError('Negative TTF')
    X, meta = windows(train_df, stride=2)
    V, vm = windows(val_df)
    y, vy = meta.ttf_pressure_high.to_numpy(), vm.ttf_pressure_high.to_numpy()
    pointer_path = STATE/'champion.json'
    pointer = json.loads(pointer_path.read_text()) if pointer_path.exists() else None
    if warm_start and not pointer: raise ValueError('No champion for warm start')
    if warm_start:
        scaler = json.loads((STATE/pointer['bundle']/'scaler.json').read_text())
        model = tf.keras.models.load_model(STATE/pointer['bundle']/'model.keras')
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='huber')
    else:
        scaler = fit_scaler(train_df, y)
        model = tf.keras.Sequential([
            tf.keras.layers.Input((SEQ_LEN, len(FEATURES))),
            tf.keras.layers.LSTM(32), tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1)])
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='huber')
    if len(X)>limit:
        # Deterministic sample across time, not just the first few events.
        idx=np.linspace(0,len(X)-1,limit,dtype=int)
        X,y,meta=X[idx],y[idx],meta.iloc[idx].reset_index(drop=True)
    sx, sv = scale_x(X,scaler), scale_x(V,scaler)
    sy, syv = scale_y(y,scaler), scale_y(vy,scaler)
    counts=meta.event_id.value_counts()
    weights=np.array([len(meta)/(len(counts)*counts[e]) for e in meta.event_id],dtype='float32')
    train_ds=tf.data.Dataset.from_tensor_slices((sx,sy,weights)).shuffle(len(sx),seed=SEED).batch(128)
    val_ds=tf.data.Dataset.from_tensor_slices((sv,syv)).batch(128)
    opts=tf.data.Options();opts.threading.private_threadpool_size=2
    train_ds=train_ds.with_options(opts);val_ds=val_ds.with_options(opts)
    baseline_value=float(pointer['baseline_ttf']) if warm_start else float(np.median(y))
    baseline=metrics(vy,np.full(len(vy),baseline_value))
    comparison=baseline
    if pointer:
        incumbent=tf.keras.models.load_model(STATE/pointer['bundle']/'model.keras')
        inc_scaler=json.loads((STATE/pointer['bundle']/'scaler.json').read_text())
        comparison=metrics(vy,inverse_y(incumbent(scale_x(V,inc_scaler),training=False).numpy().ravel(),inc_scaler))
    checkpoint=NonregressionCheckpoint(sv,vy,scaler,baseline,comparison)
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=4,restore_best_weights=True)]
    if warm_start and training_focus in {'maintenance_focus','automatic_pool'}:callbacks.append(checkpoint)
    started=time.monotonic()
    with mlflow.start_run(run_name='warm-start' if warm_start else 'base-lstm') as run:
        h=model.fit(train_ds,validation_data=val_ds,epochs=epochs,verbose=2,
                    callbacks=callbacks)
        if checkpoint.best_weights is not None:model.set_weights(checkpoint.best_weights)
        predictions=inverse_y(model(sv,training=False).numpy().ravel(),scaler)
        score=metrics(vy,predictions)
        gate=passes_gate(score,baseline,comparison)
        bundle_name=f'bundles/{run.info.run_id}'
        bundle=STATE/bundle_name;bundle.mkdir(parents=True,exist_ok=True)
        model.save(bundle/'model.keras');write_json(bundle/'scaler.json',scaler)
        report={'run_id':run.info.run_id,'bundle':bundle_name,'validation':score,
                'baseline_validation':baseline,'incumbent_validation':comparison,'gate_passed':gate,
                'gate_policy':'validation MAE, RMSE and log-MAE <= training-median baseline and incumbent',
                'baseline_ttf':baseline_value,'mode':'warm_start' if warm_start else 'scratch',
                'training_dataset':__import__('pathlib').Path(training_csv).name if training_csv else '08_M01_pressure_high_train.csv',
                'training_data_sha256':hashlib.sha256((__import__('pathlib').Path(training_csv) if training_csv else DATA/'08_M01_pressure_high_train.csv').read_bytes()).hexdigest(),
                'training_focus':training_focus,'selected_epoch':checkpoint.best_epoch,
                'epochs_completed':len(h.history['loss']),'train_sequences':len(X),'validation_sequences':len(V),
                'seconds':time.monotonic()-started,'time_unit':'source_time_unit',
                'research_demo_only':True,'history':h.history}
        mlflow.log_params({'seq_len':SEQ_LEN,'n_features':len(FEATURES),'epochs':epochs,
                           'seed':SEED,'mode':report['mode'],'target':'TTF_pressure_high','training_focus':training_focus,
                           'train_stride':2,'target_transform':'standardized_log1p',
                           'feature_order':','.join(FEATURES),
                           'training_data_sha256':hashlib.sha256((DATA/'08_M01_pressure_high_train.csv' if training_csv is None else __import__('pathlib').Path(training_csv)).read_bytes()).hexdigest()})
        for k,v in score.items():mlflow.log_metric('validation_'+k,v)
        for k,v in baseline.items():mlflow.log_metric('baseline_validation_'+k,v)
        mlflow.log_metric('gate_passed',int(gate))
        mlflow.log_artifact(str(bundle/'scaler.json'))
        mlflow.log_artifact(str(DATA/'dataset_info.json'))
        logged=mlflow.tensorflow.log_model(model,name='model',input_example=sx[:1])
        version=mlflow.register_model(logged.model_uri,MODEL_NAME)
        report['version']=str(version.version)
        report['registered_at']=datetime.fromtimestamp(version.creation_timestamp/1000,timezone.utc).isoformat()
        client=MlflowClient()
        client.set_model_version_tag(MODEL_NAME,version.version,'gate_passed',str(gate).lower())
        client.set_model_version_tag(MODEL_NAME,version.version,'bundle',bundle_name)
        if gate:
            client.set_registered_model_alias(MODEL_NAME,'champion',version.version)
            # Atomic pointer: model and frozen scaler swap together.
            if pointer:write_json(STATE/'previous_champion.json',pointer)
            write_json(pointer_path,report)
        write_json(bundle/'report.json',report);write_json(STATE/'latest_training.json',report)
        mlflow.log_dict(report,'training_report.json')
        print('[GATE PASSED]' if gate else '[GATE FAILED]',json.dumps({k:v for k,v in report.items() if k!='history'}),flush=True)
        return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--epochs',type=int,default=15)
    parser.add_argument('--warm-start',action='store_true');parser.add_argument('--limit',type=int,default=15000)
    args=parser.parse_args();train(args.epochs,args.warm_start,args.limit)
