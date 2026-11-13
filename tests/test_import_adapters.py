import sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]))
import numpy as np
import pandas as pd
from recording_selection import detect_recording_blocks
from source_adapter import (timestamp_unit_guidance, gaze_rows_only,
    interpret_validity_flags,paired_zero_gaze_signals,interpret_paired_zero_gaze)
from cirruspp_core import timestamp_diagnostics, numeric_signal_columns

class ImportAdaptersTest(unittest.TestCase):
    def test_prefer_recording_name_over_participant(self):
        df=pd.DataFrame({'Participant name':['A']*80+['B']*40,
                         'Recording name':['one']*40+['two']*40+['three']*40,
                         'Recording timestamp':list(range(40))*3})
        blocks=detect_recording_blocks(df)
        self.assertEqual(len(blocks),3)
        self.assertEqual(blocks[0].source_column,'Recording name')
    def test_tobii_metadata_unit_verification(self):
        df=pd.DataFrame({'Recording timestamp':np.arange(200)*8333,
                         'Recording duration':[int(199*8333/1000)]*200,
                         'Gaze point X':np.ones(200)})
        r=timestamp_unit_guidance(df,'Recording timestamp')
        self.assertEqual(r['unit'],'Microseconds (µs)')
        self.assertEqual(r['confidence'],'verified')
    def test_no_guess_on_generic_numeric_clock(self):
        df=pd.DataFrame({'time':np.arange(100)*1000,'temperature':np.ones(100)})
        self.assertIsNone(timestamp_unit_guidance(df,'time'))
    def test_eye_tracker_rows_remove_event_ties_without_mutating_source(self):
        df=pd.DataFrame({'Sensor':['Eye Tracker','Mouse','Eye Tracker'],
                         'Recording timestamp':[100,100,101],
                         'Gaze point X':[1,np.nan,2]})
        filtered,log=gaze_rows_only(df)
        self.assertEqual(len(filtered),2)
        self.assertEqual(log['excluded_rows'],1)
        self.assertEqual(len(df),3)
        self.assertEqual(filtered['Recording timestamp'].tolist(),[100,101])
    def test_explicit_missing_flag_masks_only_selected_signals(self):
        df=pd.DataFrame({'time':[0,1,2], 'x':[20,-32768,25], 'y':[2,-32768,4],
                         'missing':[0,1,0]})
        altered,meta=interpret_validity_flags(df,['x','y'],missing_flag=True)
        self.assertTrue(np.isnan(altered.loc[1,'x']))
        self.assertTrue(np.isnan(altered.loc[1,'y']))
        self.assertEqual(altered.time.tolist(),[0,1,2])
        self.assertEqual(df.loc[1,'x'],-32768)
        self.assertEqual(len(meta['changes']),2)
        self.assertNotIn('missing',numeric_signal_columns(df))
    def test_paired_zero_is_opt_in(self):
        df=pd.DataFrame({'gaze_x_left':[0,0,10], 'gaze_y_left':[0,4,5]})
        self.assertEqual(paired_zero_gaze_signals(df)[0][2],1)
        result,changes=interpret_paired_zero_gaze(df,['gaze_x_left','gaze_y_left'])
        self.assertTrue(np.isnan(result.loc[0,'gaze_x_left']))
        self.assertEqual(result.loc[1,'gaze_x_left'],0)
        self.assertEqual(df.loc[0,'gaze_x_left'],0)
    def test_quantised_clock_rate_not_inverse_median(self):
        ticks=np.cumsum(np.tile([1,1,2,1,2,1,2],4000))
        d=timestamp_diagnostics(pd.Series(ticks/1000))
        self.assertTrue(650 < d['sampling_rate_hz'] < 900)
        self.assertIn('quantised',d['sampling_rate_estimation'])

if __name__=='__main__':unittest.main()
