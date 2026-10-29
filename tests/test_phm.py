import unittest
import numpy as np
from phm.features import frame,windows,scale_y,inverse_y,fit_scaler
from phm.config import FEATURES


class SequenceRules(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df=frame('train')
        cls.x,cls.meta=windows(cls.df,max_windows=1)
        m=cls.meta.iloc[0]
        cls.sequence=cls.df[(cls.df.segment_id==m.segment_id)&(cls.df.time_bin<=m.time_bin)].tail(20).copy()

    def test_target_belongs_to_last_input_not_next_row(self):
        x,m=windows(self.sequence)
        self.assertEqual(m.iloc[0].time,self.sequence.iloc[-1].time)
        self.assertEqual(m.iloc[0].ttf_pressure_high,self.sequence.iloc[-1].ttf_pressure_high)
        np.testing.assert_allclose(x[0,-1],self.sequence.iloc[-1][FEATURES].to_numpy(dtype=float),rtol=1e-6)

    def test_time_gap_cannot_be_joined(self):
        bad=self.sequence.copy();bad.loc[bad.index[10],'time_bin']+=2
        with self.assertRaises(ValueError):windows(bad)

    def test_event_boundary_cannot_be_joined(self):
        bad=self.sequence.copy();bad.loc[bad.index[10]:,'event_id']+=1
        with self.assertRaises(ValueError):windows(bad)

    def test_missing_input_cannot_be_skipped(self):
        bad=self.sequence.copy();bad.loc[bad.index[10],FEATURES[0]]=np.nan
        with self.assertRaises(ValueError):windows(bad)

    def test_splits_have_no_shared_event_or_timestamp(self):
        validation,test=frame('validation'),frame('test')
        for a,b in [(self.df,validation),(self.df,test),(validation,test)]:
            self.assertFalse(set(a.time)&set(b.time));self.assertFalse(set(a.event_id)&set(b.event_id))

    def test_target_scaler_round_trip(self):
        y=np.array([0.,100.,1000000.]);s=fit_scaler(self.df,y)
        np.testing.assert_allclose(inverse_y(scale_y(y,s),s),y,rtol=1e-5,atol=1e-5)


if __name__=='__main__':unittest.main()
