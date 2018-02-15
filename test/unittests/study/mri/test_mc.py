from nipype import config
from mbianalysis.study.mri.structural.t1 import CoregisteredT1Study
from mbianalysis.study.mri.structural.t2 import CoregisteredT2Study
config.enable_debug_mode()
from nianalysis.dataset import Dataset, Field  # @IgnorePep8
from nianalysis.data_formats import nifti_gz_format, dicom_format  # @IgnorePep8
from mbianalysis.study.mri.epi import CoregisteredEPIStudy  # @IgnorePep8
from mbianalysis.study.mri.base import MRIStudy
from mbianalysis.testing import BaseTestCase as TestCase  # @IgnorePep8 @Reimport
from mbianalysis.study.mri.structural.diffusion_coreg import CoregisteredDWIStudy  # @IgnorePep8
from mbianalysis.study.mri.motion_detection import MotionDetectionStudy

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

#     def test_t1_mc(self):
#         study = self.create_study(
#             CoregisteredT1Study, 't1_reg_study', inputs={
#                 't1': Dataset('t1', nifti_gz_format),
#                 'reference': Dataset('reference', nifti_gz_format)})
#         study.t1_motion_mat_pipeline().run(work_dir=self.work_dir)
#         self.assertDatasetCreated('t1_motion_mats', study.name)

#     def test_hd_info_extraction(self):
#         study = self.create_study(
#             MRIStudy, 't2_reg_study', inputs={
#                 'dicom_file': Dataset('dwi', dicom_format)})
#         study.header_info_extraction_pipeline().run(work_dir=self.work_dir)
#         self.assertField('tr', 2.45, study.name)
#         self.assertField('start_time', '124629.127500', study.name)
#         self.assertField('tot_duration', '602', study.name)
#         self.assertField('real_duration', '592.9', study.name)
#         self.assertField('ped', '', study.name)
#         self.assertField('phase_offset', '', study.name)

    def test_dwi_mc(self):
        study = self.create_study(
            MotionDetectionStudy, 'mc_detection_study', inputs={
                'epi1': Dataset('epi', nifti_gz_format),
                'epi2': Dataset('epi', nifti_gz_format),
                'reference': Dataset('reference', nifti_gz_format)})
        study.epi1_motion_mat_pipeline().run(work_dir=self.work_dir)
        self.assertDatasetCreated('epi1_motion_mats', study.name)
#     def test_t2_mc(self):
#         study = self.create_study(
#             CoregisteredT2Study, 't2_reg_study', inputs={
#                 't2': Dataset('pd', nifti_gz_format),
#                 'reference': Dataset('ref_2', nifti_gz_format)})
#         study.t2_motion_mat_pipeline().run(work_dir=self.work_dir)
#         self.assertDatasetCreated('t2_motion_mats', study.name)
