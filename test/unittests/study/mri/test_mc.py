from nipype import config
from nianalysis.study.mri.structural.t1 import CoregisteredT1Study
config.enable_debug_mode()
from nianalysis.dataset import Dataset  # @IgnorePep8
from nianalysis.data_formats import nifti_gz_format  # @IgnorePep8
from nianalysis.study.mri.epi import CoregisteredEPIStudy  # @IgnorePep8
from nianalysis.study.mri.base import MRIStudy
from nianalysis.testing import BaseTestCase as TestCase  # @IgnorePep8 @Reimport


class TestMC(TestCase):

#     def test_epi_mc(self):
#         study = self.create_study(
#             CoregisteredEPIStudy, 'epi_reg_study', inputs={
#                 'epi': Dataset('epi', nifti_gz_format),
#                 'reference': Dataset('reference', nifti_gz_format)})
#         study.epi_motion_mat_pipeline().run(work_dir=self.work_dir)
#         self.assertDatasetCreated('epi_motion_mats', study.name)
#         self.assertField('tr', 2.45, study.name)
#         self.assertField('start_time', '124629.127500', study.name)
#         self.assertField('tot_duration', '602', study.name)
#         self.assertField('real_duration', '592.9', study.name)
#         self.assertField('ped', '', study.name)
#         self.assertField('phase_offset', '', study.name)

    def test_t1_mc(self):
        study = self.create_study(
            CoregisteredT1Study, 't1_reg_study', inputs={
                't1': Dataset('t1', nifti_gz_format),
                'reference': Dataset('reference', nifti_gz_format)})
        study.t1_motion_mat_pipeline().run(work_dir=self.work_dir)
        self.assertDatasetCreated('t1_motion_mats', study.name)
