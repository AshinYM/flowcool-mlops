"""Read all 20 training machines sequentially; retain causal windows and frozen splits."""
from pathlib import Path
import argparse
import gc
import json
import time
import numpy as np
import pandas as pd
from phm.config import ROOT, FEATURES, SEQ_LEN

SOURCE=ROOT.parent/'phm_data_challenge_2018'
DEST=ROOT/'data/phm2018_fleet'
FAULT='Flowcool Pressure Too High Check Flowcool Pump'
TARGET='TTF_'+FAULT
# Chosen before any fleet model evaluation, by event availability, not model scores.
FINAL_TOOLS={'01M02','09M02'}
CONTEXT=['stage','recipe','recipe_step','FIXTURESHUTTERPOSITION']


def indices(df, stride=1):
    finite=np.isfinite(df[FEATURES].to_numpy()).all(axis=1)
    boundary=(df.Tool.ne(df.Tool.shift()) | df.segment_id.ne(df.segment_id.shift())
              | df.event_id.ne(df.event_id.shift()) | df.time_bin.diff().ne(1)
              | df['split'].ne(df['split'].shift()) | pd.Series(finite).ne(pd.Series(finite).shift()))
    starts=np.flatnonzero(boundary.to_numpy());ends=np.r_[starts[1:],len(df)]
    parts=[np.arange(a+SEQ_LEN-1,b,stride,dtype=np.int64) for a,b in zip(starts,ends)
           if b-a>=SEQ_LEN and finite[a]]
    return np.concatenate(parts) if parts else np.array([],dtype=np.int64)


def main():
    begin=time.monotonic();DEST.mkdir(exist_ok=True);(DEST/'examples').mkdir(exist_ok=True)
    summary=[];split_files={s:[] for s in ['train','validation','test']}
    paths=sorted((SOURCE/'train').glob('*_DC_train.csv'))
    for number,path in enumerate(paths):
        started=time.monotonic();name=path.stem.removesuffix('_DC_train');tool=name.replace('_','')
        df=pd.read_csv(path,usecols=['time','Tool']+CONTEXT+FEATURES[:-1])
        labels=pd.read_csv(SOURCE/'train/train_ttf'/path.name,usecols=['time',TARGET])
        assert np.array_equal(df.time,labels.time)
        raw_rows=len(df);df['ttf_pressure_high']=labels[TARGET].to_numpy();del labels
        df=df.drop_duplicates('time',keep='last').reset_index(drop=True)
        faults=pd.read_csv(SOURCE/'train/train_faults'/f'{name}_train_fault_data.csv').drop_duplicates()
        events=np.sort(faults.loc[faults.fault_name.eq(FAULT),'time'].unique())
        stops=np.searchsorted(np.sort(faults.time.unique()),df.time,side='right')
        boundary=(df.time.diff().gt(60)|df.stage.ne(df.stage.shift())|df.recipe.ne(df.recipe.shift())
                  |pd.Series(stops).ne(pd.Series(stops).shift()))
        df['source_segment']=boundary.cumsum();df['time_bin']=df.time//60
        group=df.groupby(['source_segment','time_bin'],sort=True)
        agg=group[FEATURES[:-1]].mean();agg['FLOWCOOLPRESSURE_MAX']=group.FLOWCOOLPRESSURE.max()
        agg['time']=group.time.max();agg['sample_count']=group.size()
        last=group.tail(1).set_index(['source_segment','time_bin'])
        for c in ['Tool','ttf_pressure_high']+CONTEXT:agg[c]=last[c]
        agg=agg.reset_index();del group,last,df;gc.collect()
        next_idx=np.searchsorted(events,agg.time,side='left');labeled=np.isfinite(agg.ttf_pressure_high)
        if labeled.any():
            assert (next_idx[labeled]<len(events)).all()
            assert np.allclose(agg.loc[labeled,'ttf_pressure_high'],events[next_idx[labeled]]-agg.loc[labeled,'time'])
        # An entire future-event interval stays together; holdout equipment never trains.
        cutoff=max(1,int(len(events)*0.8))
        agg['split']=np.where(~labeled,'unlabeled',np.where(tool in FINAL_TOOLS,'test',
                                          np.where(next_idx<cutoff,'train','validation')))
        agg['event_id']=number*1000+next_idx
        boundary=agg.source_segment.ne(agg.source_segment.shift())|agg.time_bin.diff().ne(1)
        agg['segment_id']=number*1_000_000+boundary.cumsum()
        agg['usable_for_training']=labeled & np.isfinite(agg[FEATURES]).all(axis=1)
        agg.to_csv(DEST/f'{name}_pressure_high.csv',index=False,float_format='%.9g')
        agg.to_parquet(DEST/f'{name}.parquet',index=False)
        ix=indices(agg)
        columns=['time','time_bin','segment_id','Tool']+FEATURES
        if len(ix):
            for suffix,end in [('first',ix[0]),('latest',ix[-1])]:
                payload={'sequence':json.loads(agg.iloc[end-19:end+1][columns].to_json(orient='records'))}
                (DEST/'examples'/f'{tool}_{suffix}.json').write_text(json.dumps(payload))
        counts={}
        for split in split_files:
            part=agg[agg['split'].eq(split)&agg.usable_for_training].reset_index(drop=True)
            p=DEST/f'{name}_{split}.parquet';part.to_parquet(p,index=False);split_files[split].append(p)
            counts[split]={'rows':len(part),'windows':len(indices(part))}
            if split=='train' and tool=='08M01':
                # A small, genuine labeled upload for the operational warm-start demo.
                sample=part.tail(12000).copy();sample.to_csv(DEST/'demo_train.csv',index=False)
            if split=='validation' and tool=='08M01':
                part.tail(12000).to_csv(DEST/'demo_validation.csv',index=False)
        row={'tool':tool,'source_rows':raw_rows,'aggregated_rows':len(agg),'fault_events':len(events),
             'role':'unseen_equipment_final_test' if tool in FINAL_TOOLS else 'chronological_train_validation',
             'splits':counts,'seconds':round(time.monotonic()-started,2)}
        summary.append(row);print(json.dumps(row),flush=True);del agg,part;gc.collect()
    for split,files in split_files.items():
        parts=[pd.read_parquet(p) for p in files];full=pd.concat(parts,ignore_index=True)
        full.to_parquet(DEST/f'{split}.parquet',index=False);del parts,full;gc.collect()
    report={'profile':'fleet','tools':[x['tool'] for x in summary],
            'final_test_tools':sorted(FINAL_TOOLS),'time_unit':'source_time_unit',
            'source_rows':sum(x['source_rows'] for x in summary),
            'output_rows':sum(x['aggregated_rows'] for x in summary),
            'fault_events':sum(x['fault_events'] for x in summary),
            'split_windows':{s:sum(x['splits'][s]['windows'] for x in summary) for s in split_files},
            'split_rows':{s:sum(x['splits'][s]['rows'] for x in summary) for s in split_files},
            'features':FEATURES,'bin_width':60,'seq_len':SEQ_LEN,'equipment':summary,
            'seconds':round(time.monotonic()-begin,2),
            'policy':'All 20 training machines processed. Two untouched equipment holdouts. Other equipment: chronological 80/20 event intervals. Unlabeled rows retained, never fabricated.',
            'official_test':'Five official test CSVs have no TTF labels; no claimed accuracy on these files.'}
    (DEST/'dataset_info.json').write_text(json.dumps(report,indent=2))
    print('DONE',json.dumps({k:v for k,v in report.items() if k!='equipment'}),flush=True)


if __name__=='__main__':main()
