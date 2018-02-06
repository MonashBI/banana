from nipype import config
config.enable_debug_mode()
from nianalysis.dataset import Dataset  # @IgnorePep8
from nianalysis.data_formats import nifti_gz_format  # @IgnorePep8
from nianalysis.study.mri.epi import CoregisteredEPIStudy  # @IgnorePep8
from nianalysis.study.mri.base import MRIStudy
from nianalysis.testing import BaseTestCase as TestCase  # @IgnorePep8 @Reimport


class TestMC(TestCase):

    def test_brain_mask(self):
        study = self.create_study(
            MRIStudy, 'segmentation_study', input_datasets={
                'ref_brain': Dataset('ref_brain', nifti_gz_format)})
        study.segmentation_pipeline().run(work_dir=self.work_dir)
        self.assertDatasetCreated('wm_seg.nii.gz', study.name)
