"""Automatic selection must rotate real, labeled train cohorts without holdout."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from fastapi import BackgroundTasks
from phm import app as service, training_pool


class AutomaticCohorts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory()
        with patch.object(training_pool,'STATE',Path(cls.temp.name)):
            cls.pool=training_pool.prepare(service.inspect_dataset)

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_every_cohort_is_real_train_and_excludes_holdout(self):
        original=pd.read_csv(service.DATA/'08_M01_pressure_high_train.csv',float_precision='round_trip').set_index('time')
        hashes=[]
        for item in self.pool:
            self.assertTrue(item['retrain_eligible']);self.assertFalse(item['holdout_overlap'])
            rows=pd.read_csv(item['path'],float_precision='round_trip').set_index('time')
            self.assertTrue(rows['split'].eq('train').all())
            self.assertTrue(rows.index.isin(original.index).all())
            np.testing.assert_allclose(rows[service.FEATURES+['ttf_pressure_high']],original.loc[rows.index,service.FEATURES+['ttf_pressure_high']],rtol=1e-12)
            hashes.append(item['data_sha256'])
        self.assertEqual(len(hashes),len(set(hashes)))

    def test_cursor_rotates_and_survives_reload(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(training_pool,'STATE',Path(folder)):
            self.assertEqual(training_pool.cursor(),0)
            training_pool.advance(0)
            self.assertEqual(training_pool.cursor(),1)
            self.assertEqual(training_pool.status(self.pool)['next_dataset_name'],self.pool[1]['name'])

    def test_alert_without_dataset_id_queues_automatic_training(self):
        points=service.example()
        req=service.Batch(sequences=[points])
        with patch.object(service.runtime,'predict',return_value=(np.array([123.]),{'version':'1'})),patch.object(service.runtime,'monitor',return_value={'status':'input_shift'}),patch.object(service,'queue_job',return_value={'job_id':'verified'}) as queue:
            result=service.batch(req,BackgroundTasks())
            self.assertEqual(result['response_job']['job_id'],'verified')
            self.assertEqual(queue.call_args.args[0].upload_id,'automatic')
            service.batch(service.Batch(sequences=[points],auto_response=False),BackgroundTasks())
            self.assertEqual(queue.call_count,1)

    def test_busy_job_does_not_consume_next_cohort(self):
        with patch.object(service,'AUTO_POOL',self.pool),patch.object(service.runtime,'pointer',return_value={'version':'1'}),patch.object(training_pool,'advance') as advance:
            service.JOB_LOCK.acquire()
            try:
                with self.assertRaises(service.HTTPException) as caught:
                    service.queue_job(service.Retrain(upload_id='automatic'),BackgroundTasks())
                self.assertEqual(caught.exception.status_code,409);advance.assert_not_called()
            finally:service.JOB_LOCK.release()

if __name__=='__main__':unittest.main()
