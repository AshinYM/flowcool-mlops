"""Frozen equipment holdout diagnostic. A rejected candidate remains rejected."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','2')
import json
import numpy as np
import tensorflow as tf
import mlflow
from phm.config import STATE, DATA, FEATURES, PROFILE, SEQ_LEN
from phm.features import frame, inverse_y, metrics
from phm.fleet_train import Batches
from phm.evaluate import choose, alert_metrics, HORIZON
from phm.train import setup_tracking, write_json


def pressure_scores(batches, scaler):
    column=FEATURES.index('FLOWCOOLPRESSURE_MAX')
    values=[]
    for x,_ in batches.generator():
        values.append((x[:,:,column]*scaler['std'][column]+scaler['mean'][column]).max(axis=1))
    return np.concatenate(values)


def main():
    if PROFILE!='fleet':raise ValueError('Set PHM_PROFILE=fleet')
    # Selection is finalized using validation, before touching this final test.
    report=json.loads((STATE/'latest_training.json').read_text())
    previous=STATE/'evaluation.json'
    if previous.exists():
        saved=json.loads(previous.read_text())
        if saved.get('run_id')==report['run_id']:
            print('Already evaluated this frozen candidate; reusing saved result.');return saved
        raise ValueError('Final holdout already inspected; do not tune and repeatedly select on it')
    bundle=STATE/report['bundle'];scaler=json.loads((bundle/'scaler.json').read_text())
    model=tf.keras.models.load_model(bundle/'model.keras')
    va=Batches(frame('validation'),scaler);te=Batches(frame('test'),scaler)
    vp=inverse_y(va.predict(model),scaler);tp=inverse_y(te.predict(model),scaler)
    vt,vm=choose(va.meta,vp,True)
    vp_pressure=pressure_scores(va,scaler);tp_pressure=pressure_scores(te,scaler)
    pt,pm=choose(va.meta,vp_pressure,False)
    score=metrics(te.actual,tp)
    baseline=metrics(te.actual,np.full(len(tp),report['baseline_ttf']))
    info=json.loads((DATA/'dataset_info.json').read_text())
    result={'run_id':report['run_id'],'model_version':report['version'],
        'model_role':'champion' if report['gate_passed'] else 'rejected_candidate_diagnostic',
        'validation_gate_passed':report['gate_passed'],
        'deployment_changed_by_this_evaluation':False,
        'time_unit':'source_time_unit','horizon':HORIZON,
        'horizon_basis':'demonstration scenario in anonymous source units; not seconds/hours',
        'test':score,'test_constant_baseline':baseline,
        'beats_baseline':all(score[k]<baseline[k] for k in score),
        'test_sequences':len(te.end),'test_tools':info['final_test_tools'],
        'threshold_selection':'validation only: event recall minus 0.01 * false episodes',
        'model_ttf_alert_threshold':vt,'pressure_alert_threshold':pt,
        'validation_alerts':{'lstm':vm,'pressure':pm},
        'test_alerts':{'lstm':alert_metrics(te.meta,tp<=vt),
                       'pressure':alert_metrics(te.meta,tp_pressure>=pt)},
        'test_per_tool':{tool:metrics(te.actual[idx],tp[idx]) for tool,idx in te.meta.groupby('Tool').groups.items()},
        'commercial_effectiveness_proven':False,
        'limitations':['Only two previously unseen equipment holdouts',
            'TTF is maintenance-stop time, not verified physical failure onset',
            'Different alert budgets; comparisons do not establish business superiority',
            'Sensor gaps exclude windows; no claim about continuous operating uptime',
            'Anonymous units; no economic labels or real-time factory connection',
            'Validation gate failure is not overruled by the final diagnostic']}
    predictions=te.meta.copy();predictions['predicted_ttf']=tp
    predictions['lstm_alert']=tp<=vt;predictions['pressure_alert']=tp_pressure>=pt
    predictions.to_csv(STATE/'test_predictions.csv',index=False)
    write_json(previous,result)
    setup_tracking()
    with mlflow.start_run(run_name='fleet-frozen-equipment-diagnostic'):
        mlflow.set_tags({'model_version':report['version'],'model_role':result['model_role']})
        for k,v in score.items():mlflow.log_metric('test_'+k,v)
        for k,v in baseline.items():mlflow.log_metric('test_baseline_'+k,v)
        mlflow.log_dict(result,'evaluation.json')
        mlflow.log_artifact(str(STATE/'test_predictions.csv'))
    print(json.dumps(result,ensure_ascii=False,indent=2));return result


if __name__=='__main__':main()
