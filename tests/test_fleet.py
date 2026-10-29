"""Boundary/holdout/contract tests without constructing millions of tensors."""
import json
import unittest
import tempfile
from unittest.mock import patch
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi import HTTPException
from phm.config import PROFILE, DATA, STATE, FEATURES
from phm.prepare_fleet import indices
from phm.features import windows
from phm.train import passes_gate
from phm import runtime


@unittest.skipUnless(PROFILE=='fleet','Run with PHM_PROFILE=fleet')
class FleetRules(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from phm.app import Prediction,predict
        class Client:
            def post(self,path,**kwargs):
                from types import SimpleNamespace
                try:
                    req=Prediction.model_validate(kwargs['json'])
                    predict(req)
                    return SimpleNamespace(status_code=200)
                except ValueError:return SimpleNamespace(status_code=422)
                except HTTPException as e:return SimpleNamespace(status_code=e.status_code)
        cls.client=Client()
        cls.example=json.loads((DATA/'examples/08M01_first.json').read_text())
        cls.df=pd.DataFrame(cls.example['sequence'])
        cls.df['event_id']=8001;cls.df['split']='train';cls.df['ttf_pressure_high']=100.

    def test_index_generator_matches_materialized_windows(self):
        ends=indices(self.df);x,m=windows(self.df)
        self.assertEqual(len(ends),len(x));self.assertEqual(int(ends[0]),19)
        np.testing.assert_array_equal(self.df.iloc[ends].time,m.time)

    def test_mixed_tools_never_joined(self):
        bad=self.df.copy();bad.loc[10:,'Tool']='09M01'
        self.assertEqual(len(indices(bad)),0)
        request=json.loads(json.dumps(self.example));request['sequence'][10]['Tool']='09M01'
        self.assertEqual(self.client.post('/predict',json=request).status_code,422)

    def test_required_sensor_and_output_contract(self):
        req=json.loads(json.dumps(self.example));del req['sequence'][0][FEATURES[-1]]
        self.assertEqual(self.client.post('/predict',json=req).status_code,422)
        from phm.app import PredictionResponse
        with self.assertRaises(ValueError):
            PredictionResponse(tool='08M01',as_of=1,predicted_ttf=float('nan'),
                time_unit='source_time_unit',model_version='2',usage='test',reliability_warning='test',monitoring={})

    def test_whole_equipment_holdouts_and_event_disjointness(self):
        keys={s:pd.read_parquet(DATA/f'{s}.parquet',columns=['Tool','event_id']) for s in ['train','validation','test']}
        self.assertEqual(set(keys['test'].Tool),{'01M02','09M02'})
        for a,b in [('train','validation'),('train','test'),('validation','test')]:
            self.assertFalse(set(keys[a].event_id)&set(keys[b].event_id))
        self.assertFalse(set(keys['test'].Tool)&set(keys['train'].Tool))

    def test_gate_failure_cannot_create_champion(self):
        baseline=dict(mae=1.,rmse=1.,log_mae=1.)
        self.assertFalse(passes_gate(dict(mae=1.,rmse=2.,log_mae=1.),baseline,baseline))
        self.assertFalse(passes_gate(dict(mae=float('nan'),rmse=1.,log_mae=1.),baseline,baseline))
        latest=json.loads((STATE/'latest_training.json').read_text())
        if not latest['gate_passed']:
            self.assertFalse((STATE/'champion.json').exists())
            self.assertEqual(self.client.post('/predict',json=self.example).status_code,503)

    def test_client_validation_errors_are_not_server_failures(self):
        saved=list(runtime.REQUESTS['recent']);runtime.REQUESTS['recent'].clear()
        try:
            with tempfile.TemporaryDirectory() as directory, patch.object(runtime,'STATE',Path(directory)):
                for _ in range(21):runtime.request_record('/predict','POST',422,2.)
                m=runtime.service_metrics()
                self.assertEqual(m['error_rate_5xx'],0);self.assertFalse(m['alert'])
                runtime.request_record('/predict','POST',500,2.)
                runtime.request_record('/predict','POST',500,2.)
                self.assertTrue(runtime.service_metrics()['alert'])
        finally:
            runtime.REQUESTS['recent'].clear();runtime.REQUESTS['recent'].extend(saved)


if __name__=='__main__':unittest.main()
