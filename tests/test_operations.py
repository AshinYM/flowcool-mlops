import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from phm.app import app
from phm import runtime, operations


class OperationsRules(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory()
        self.state_patch=patch.object(runtime,'STATE',Path(self.directory.name))
        self.state_patch.start()

    def tearDown(self):
        self.state_patch.stop()
        self.directory.cleanup()

    def test_polling_cannot_dilute_error_rate(self):
        old=list(runtime.REQUESTS['recent'])
        try:
            runtime.REQUESTS['recent'].clear()
            client=TestClient(app)
            for _ in range(3):
                self.assertEqual(client.get('/metrics/summary').status_code,200)
                self.assertEqual(client.get('/operations/summary').status_code,200)
            self.assertEqual(len(runtime.REQUESTS['recent']),0)
            self.assertEqual(client.post('/predict',json={'sequence':[]}).status_code,422)
            self.assertEqual(len(runtime.REQUESTS['recent']),1)
            self.assertEqual(runtime.service_metrics()['validation_errors_4xx'],1)
            self.assertEqual(runtime.service_metrics()['error_rate_5xx'],0)
        finally:
            runtime.REQUESTS['recent'].clear();runtime.REQUESTS['recent'].extend(old)

    def test_rejected_candidate_does_not_look_deployed(self):
        with tempfile.TemporaryDirectory() as temp:
            state=Path(temp)
            for version,passed in [('1',True),('2',False)]:
                bundle=state/'bundles'/version;bundle.mkdir(parents=True)
                (bundle/'report.json').write_text(json.dumps({'version':version,'gate_passed':passed,
                                                            'validation':{'rmse':1 if passed else 2}}))
            event={'at':'2026-10-08T00:00:00+00:00','job_id':'job'}
            (state/'aiops.jsonl').write_text('\n'.join(json.dumps({**event,'kind':k}) for k in
                                                       ['retrain_triggered','candidate_rejected']))
            with patch.object(operations,'STATE',state),patch.object(runtime,'pointer',return_value={'version':'1'}):
                d=operations.summary({})
            self.assertEqual([r['deployment'] for r in d['model_history']],['rejected','active'])
            self.assertEqual(d['pipeline'][-2]['status'],'rejected')
            self.assertEqual(d['pipeline'][-1]['status'],'retained')

    def test_old_model_alert_cannot_trigger_current_model(self):
        saved=list(runtime.EVENTS);runtime.EVENTS.clear()
        try:
            runtime.EVENTS.append({'kind':'monitor_alert','model_version':'old','at':'2026-10-08T00:00:00+00:00'})
            with patch.object(runtime,'pointer',return_value={'version':'new'}):
                response=TestClient(app).post('/monitoring/auto-response',json={'upload_id':'missing'})
            self.assertEqual(response.status_code,409)
        finally:
            runtime.EVENTS.clear();runtime.EVENTS.extend(saved)

    def test_history_uses_actual_registry_registration_time(self):
        import sqlite3
        from phm.config import MODEL_NAME
        with tempfile.TemporaryDirectory() as temp:
            state=Path(temp);bundle=state/'bundles'/'run';bundle.mkdir(parents=True)
            (bundle/'report.json').write_text(json.dumps({'version':'1','gate_passed':True,'validation':{'rmse':1}}))
            with sqlite3.connect(state/'mlflow.db') as con:
                con.execute('CREATE TABLE model_versions(name TEXT,version INTEGER,creation_time INTEGER)')
                con.execute('INSERT INTO model_versions VALUES(?,?,?)',(MODEL_NAME,1,1791365072089))
            with patch.object(operations,'STATE',state),patch.object(runtime,'pointer',return_value={'version':'1'}):
                result=operations.summary({})
            self.assertEqual(result['model_history'][0]['registered_at'],'2026-10-07T09:24:32.089000+00:00')
            self.assertEqual(result['champion']['registered_at'],result['model_history'][0]['registered_at'])


if __name__=='__main__':unittest.main()
