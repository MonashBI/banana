from nipype import config
config.enable_debug_mode()
from mbianalysis.testing import BaseTestCase as TestCase  # @IgnorePep8 @Reimport
from mbianalysis.study.mri.motion_detection_mixin_new import (  # @IgnorePep8 @Reimport
    create_motion_detection_class)
from mbianalysis.study.multimodal.test_motion_detection import (
    MotionDetection, A, inputs)


# ref = 'reference_dicom'
# t1s = ['t1_1_dicom']
# t2s = ['t2_1_dicom']
# epis = ['epi_dicom']
# 
# MotionDetection, inputs = create_motion_detection_class(
#     'MotionDetection', ref, 't1', t1s=t1s, t2s=t2s, epis=epis)

class TestMC(TestCase):

    def test_epi_mc(self):

        study = self.create_study(
            MotionDetection, 'MotionDetection', inputs=inputs)
        study.data('motion_detection_output')
        self.assertDatasetCreated('motion_detection_output', study.name)
