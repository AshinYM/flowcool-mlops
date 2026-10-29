import json
import unittest
import numpy as np
from phm import replay
from phm.app import Prediction
from phm.features import frame,fit_scaler
from phm.config import FEATURES


class HistoricalReplayRules(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train=frame('train')
        cls.scaler=fit_scaler(cls.train,cls.train.ttf_pressure_high.to_numpy())

    def test_consecutive_replays_are_distinct_valid_real_windows(self):
        a,b=replay.next_window(self.scaler),replay.next_window(self.scaler)
        self.assertNotEqual(a['selection']['window_id'],b['selection']['window_id'])
        for result in [a,b]:
            Prediction.model_validate(result['request'])
            sequence=result['request']['sequence']
            original=self.train.set_index('time').loc[[p['time'] for p in sequence]]
            np.testing.assert_allclose([[p[f] for f in FEATURES] for p in sequence],original[FEATURES],rtol=1e-7)
            self.assertFalse(result['selection']['synthetic'])

    def test_replay_does_not_send_future_labels_to_model(self):
        result=replay.next_window(self.scaler)
        for point in result['request']['sequence']:
            self.assertNotIn('ttf_pressure_high',point)
            self.assertNotIn('event_id',point)
            self.assertNotIn('split',point)


if __name__=='__main__':unittest.main()
