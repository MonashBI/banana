from nipype import config
config.enable_debug_mode()
from banana.testing import BaseTestCase as TestCase
# from banana.study.multi.test_motion_detection import (
#     MotionDetection, inputs)
from banana.study.multi.mrpet import create_motion_correction_class


ref = 'ref'
ref_type = 't1'
t1s = ['ute']
t2s = ['t2']
epis = ['epi']
dwis = [['dwi_main', '0'], ['dwi_opposite', '-1']]


class TestMC(TestCase):

#     def test_epi_mc(self):
#  
#         study = self.create_study(
#             MotionDetection, 'MotionDetection', inputs=inputs,
#             enforce_inputs=False)
#         study.data('motion_detection_output')
#         self.assertFilesetCreated('motion_detection_output', study.name)

    def test_motion_correction(self):

        MotionCorrection, inputs, out_data = create_motion_correction_class(
            'MotionCorrection', ref, ref_type, t1s=t1s, t2s=t2s, dwis=dwis,
            epis=epis)

        study = self.create_study(
            MotionCorrection, 'MotionCorrection', inputs=inputs,
            enforce_inputs=False)
        study.data(out_data)
        self.assertFilesetCreated(out_data, study.name)
