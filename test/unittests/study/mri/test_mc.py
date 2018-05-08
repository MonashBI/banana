from nipype import config
config.enable_debug_mode()
from mbianalysis.testing import BaseTestCase as TestCase  # @IgnorePep8 @Reimport
from mbianalysis.study.multimodal.test_motion_detection import (  # @IgnorePep8 @Reimport
    MotionDetection, inputs)


class TestMC(TestCase):

    def test_epi_mc(self):

        study = self.create_study(
            MotionDetection, 'MotionDetection', inputs=inputs,
            enforce_inputs=False)
        study.data('motion_detection_output')
        self.assertDatasetCreated('motion_detection_output', study.name)
