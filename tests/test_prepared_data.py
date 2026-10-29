import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from phm import app as service, runtime


class PreparedTrainingRules(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.uploads = dict(service.UPLOADS)
        self.events = list(runtime.EVENTS)
        self.patches = [patch.object(service, 'STATE', Path(self.temp.name)),
                        patch.object(runtime, 'STATE', Path(self.temp.name)),
                        patch.dict(os.environ, {'LOADING_MODE': 'lazy'})]
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        service.UPLOADS.clear(); service.UPLOADS.update(self.uploads)
        runtime.EVENTS.clear(); runtime.EVENTS.extend(self.events)
        self.temp.cleanup()

    def test_startup_connects_confirmed_training_without_upload(self):
        with TestClient(service.app) as client:
            data = client.get('/data/status').json()['default_training']
            self.assertEqual(data['source'], 'prepared_train')
            self.assertTrue(data['confirmed_labels'])
            self.assertTrue(data['retrain_eligible'])
            self.assertFalse(data['holdout_overlap'])
            self.assertGreaterEqual(data['sequences'], 128)
            self.assertNotIn('path', data)
            self.assertEqual(service.UPLOADS[data['upload_id']]['path'],
                             str(service.DATA/'08_M01_pressure_high_train.csv'))

    def test_holdout_upload_cannot_replace_prepared_training(self):
        with TestClient(service.app) as client:
            before = client.get('/data/status').json()['default_training']
            path = service.DATA/'08_M01_pressure_high_validation.csv'
            with path.open('rb') as source:
                uploaded = client.post('/data/upload', files={'file': (path.name, source, 'text/csv')})
            self.assertEqual(uploaded.status_code, 200)
            self.assertFalse(uploaded.json()['retrain_eligible'])
            blocked = client.post('/models/retrain', json={'upload_id': uploaded.json()['upload_id']})
            self.assertEqual(blocked.status_code, 422)
            self.assertEqual(client.get('/data/status').json()['default_training'], before)

    def test_focus_data_is_unmodified_real_train_only(self):
        import pandas as pd
        import numpy as np
        with TestClient(service.app) as client:
            d=client.get('/data/status').json()['focus_training']
            self.assertTrue(d['retrain_eligible'])
            self.assertFalse(d['holdout_overlap'])
        original=pd.read_csv(service.DATA/'08_M01_pressure_high_train.csv').set_index('time')
        focus=pd.read_csv(service.DATA/'08_M01_maintenance_focus_train.csv').set_index('time')
        self.assertTrue(focus.index.isin(original.index).all())
        self.assertTrue(focus['split'].eq('train').all())
        self.assertLessEqual(focus.ttf_pressure_high.max(),86400)
        columns=service.FEATURES+['ttf_pressure_high']
        np.testing.assert_allclose(focus[columns],original.loc[focus.index,columns],rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
