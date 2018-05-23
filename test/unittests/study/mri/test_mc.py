from nipype import config
config.enable_debug_mode()
from nianalysis.testing import BaseTestCase as TestCase  # @IgnorePep8 @Reimport
# from nianalysis.study.multimodal.test_motion_detection import (  # @IgnorePep8 @Reimport
#     MotionDetection, inputs)
from nianalysis.study.multimodal.test_motion_correction import (  # @IgnorePep8 @Reimport
    MotionCorrection, inputs, out_data)


class TestMC(TestCase):

#     def test_epi_mc(self):
#  
#         study = self.create_study(
#             MotionDetection, 'MotionDetection', inputs=inputs,
#             enforce_inputs=False)
#         study.data('motion_detection_output')
#         self.assertDatasetCreated('motion_detection_output', study.name)

    def test_motion_correction(self):
 
        study = self.create_study(
            MotionCorrection, 'MotionCorrection', inputs=inputs,
            enforce_inputs=False)
        study.data(out_data)
        self.assertDatasetCreated(out_data, study.name)
