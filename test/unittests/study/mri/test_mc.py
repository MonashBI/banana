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
            CoregisteredEPIStudy, 'epi_reg_study', inputs={
                'epi': Dataset('epi', nifti_gz_format),
                'reference': Dataset('reference', nifti_gz_format),
                'ref_wmseg': Dataset('wm', nifti_gz_format)})
        study.epi_motion_mat_pipeline().run(work_dir=self.work_dir)
        self.assertDatasetCreated('epi_motion_mats', study.name)
