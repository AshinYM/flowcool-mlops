"""One final holdout evaluation; alert settings come from validation only."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
os.environ.setdefault('MLFLOW_DISABLE_AGENT_HINT','1')
os.environ.setdefault('MPLCONFIGDIR','/tmp/phm-matplotlib')
import json
import numpy as np
import pandas as pd
from phm.config import DATA, STATE, FEATURES
from phm.features import frame,windows,metrics
from phm.runtime import predict,pointer

# Scenario choice, anonymous source units; NOT hours, and NOT chosen using test.
HORIZON=86400


def alert_metrics(meta,alarm):
    ttf=meta.ttf_pressure_high.to_numpy()
    close=(ttf>=0)&(ttf<=HORIZON)
    available=set(meta.loc[close,'event_id'])
    detected=set(meta.loc[close&alarm,'event_id'])
    start=alarm & (~np.r_[False,alarm[:-1]] | meta.segment_id.ne(meta.segment_id.shift()).to_numpy())
    false_episodes=int((start & (ttf>HORIZON)).sum())
    leads={str(int(e)):float(meta.loc[(meta.event_id==e)&close&alarm,'ttf_pressure_high'].max()) for e in detected}
    return {'covered_events':len(available),'detected_events':len(detected),
            'event_recall':len(detected)/len(available) if available else None,
            'false_alert_episodes':false_episodes,'observed_windows':len(meta),
            'false_episodes_per_1000_windows':false_episodes*1000/len(meta),
            'lead_time_by_event':leads,'median_lead':float(np.median(list(leads.values()))) if leads else None}


def choose(meta,score,low):
    options=[]
    for q in [.05,.1,.2,.4,.6,.8,.9,.95]:
        threshold=float(np.quantile(score,q))
        a=score<=threshold if low else score>=threshold
        m=alert_metrics(meta,a)
        # Validation-only objective: event recall minus 1% per false episode.
        utility=(m['event_recall'] or 0)-0.01*m['false_alert_episodes']
        options.append((utility,-m['false_alert_episodes'],threshold,m))
    _,_,threshold,m=max(options,key=lambda o:(o[0],o[1]))
    return threshold,m


def main():
    p=pointer()
    V,vm=windows(frame('validation'));T,tm=windows(frame('test'))
    vp,_=predict(V);tp,_=predict(T)
    model_threshold,model_val=choose(vm,vp,True)
    pressure_index=FEATURES.index('FLOWCOOLPRESSURE_MAX')
    vpressure=V[:,:,pressure_index].max(axis=1)
    tpressure=T[:,:,pressure_index].max(axis=1)
    pressure_threshold,pressure_val=choose(vm,vpressure,False)
    model_alert=alert_metrics(tm,tp<=model_threshold)
    pressure_alert=alert_metrics(tm,tpressure>=pressure_threshold)
    baseline=metrics(tm.ttf_pressure_high,np.full(len(tm),p['baseline_ttf']))
    score=metrics(tm.ttf_pressure_high,tp)
    report={'model_version':p['version'],'time_unit':'source_time_unit',
            'horizon':HORIZON,'horizon_basis':'demonstration scenario; not seconds/hours',
            'threshold_selection':'validation only: recall - 0.01 * false alert episodes',
            'model_ttf_alert_threshold':model_threshold,'pressure_alert_threshold':pressure_threshold,
            'validation_alerts':{'lstm':model_val,'pressure':pressure_val},
            'test':score,'test_constant_baseline':baseline,
            'test_alerts':{'lstm':model_alert,'pressure':pressure_alert},
            'raw_event_split_counts':{'train':22,'validation':5,'test':5},
            'events_with_20row_windows':{'validation':int(vm.event_id.nunique()),'test':int(tm.event_id.nunique())},
            'events_without_near_fault_window_test':5-model_alert['covered_events'],
            'commercial_effectiveness_proven':False,
            'limitations':['Small validation event sample','Many pre-failure sensor gaps',
                           'Anonymous time/physical units','No maintenance cost or repair-duration labels',
                           'Alert thresholds selected separately; false-alert budgets differ',
                           'Only labeled valid windows evaluated; gaps are not counted as normal uptime']}
    result=tm[['time','event_id','segment_id','ttf_pressure_high']].copy()
    result['predicted_ttf']=tp;result['lstm_alert']=tp<=model_threshold
    result['pressure_alert']=tpressure>=pressure_threshold
    result.to_csv(STATE/'test_predictions.csv',index=False)
    per=[]
    for e,g in result.groupby('event_id'):
        row={'event_id':int(e),'windows':len(g),**metrics(g.ttf_pressure_high,g.predicted_ttf)}
        per.append(row)
    report['test_per_event']=per
    (STATE/'evaluation.json').write_text(json.dumps(report,indent=2))
    (STATE/'alert_policy.json').write_text(json.dumps({'horizon':HORIZON,'ttf_threshold':model_threshold,'pressure_threshold':pressure_threshold,'time_unit':'source_time_unit'}))
    import mlflow
    from phm.train import setup_tracking
    setup_tracking()
    with mlflow.start_run(run_name='frozen-holdout-evaluation'):
        mlflow.set_tag('model_version',p['version'])
        for k,v in score.items():mlflow.log_metric('test_'+k,v)
        mlflow.log_dict(report,'evaluation.json')
        mlflow.log_artifact(str(STATE/'test_predictions.csv'))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
