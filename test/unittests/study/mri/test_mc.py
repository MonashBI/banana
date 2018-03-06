from nipype import config
from mbianalysis.study.mri.structural.t1 import CoregisteredT1Study
from mbianalysis.study.mri.structural.t2 import CoregisteredT2Study
from nianalysis.data_formats import directory_format, text_format
config.enable_debug_mode()
from nianalysis.dataset import Dataset, Field  # @IgnorePep8
from nianalysis.data_formats import nifti_gz_format, dicom_format  # @IgnorePep8
from mbianalysis.study.mri.epi import CoregisteredEPIStudy  # @IgnorePep8
from mbianalysis.study.mri.base import MRIStudy
from mbianalysis.testing import BaseTestCase as TestCase  # @IgnorePep8 @Reimport
from mbianalysis.study.mri.structural.diffusion_coreg import CoregisteredDiffusionStudy  # @IgnorePep8
from mbianalysis.study.mri.motion_detection_metaclass import MotionDetectionStudy

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

#     def test_dwi_mc(self):
#         study = self.create_study(
#             MotionDetectionStudy, 'mc_detection_study', inputs={
#                 'epi1': Dataset('epi_dicom', dicom_format),
#                 'epi2': Dataset('epi2_dicom', dicom_format),
#                 'epi3': Dataset('epi3_dicom', dicom_format),
#                 'epi4': Dataset('epi4_dicom', dicom_format),
#                 'epi5': Dataset('epi5_dicom', dicom_format),
#                 'epi6': Dataset('epi6_dicom', dicom_format),
#                 'epi7': Dataset('epi7_dicom', dicom_format),
#                 'epi8': Dataset('epi8_dicom', dicom_format),
#                 'asl': Dataset('asl_dicom', dicom_format),
#                 't1_1': Dataset('t1_dicom', dicom_format),
#                 'ute': Dataset('ute_dicom', dicom_format),
#                 'fm': Dataset('fm_dicom', dicom_format),
#                 'reference': Dataset('reference_dicom', dicom_format),
#                 'epi1_motion_mats': Dataset('epi1_motion_mats', directory_format),
#                 'epi2_motion_mats': Dataset('epi2_motion_mats', directory_format),
#                 'epi3_motion_mats': Dataset('epi3_motion_mats', directory_format),
#                 'epi4_motion_mats': Dataset('epi4_motion_mats', directory_format),
#                 'epi5_motion_mats': Dataset('epi5_motion_mats', directory_format),
#                 'epi6_motion_mats': Dataset('epi6_motion_mats', directory_format),
#                 'epi7_motion_mats': Dataset('epi7_motion_mats', directory_format),
#                 'epi8_motion_mats': Dataset('epi8_motion_mats', directory_format),
#                 'asl_motion_mats': Dataset('asl_motion_mats', directory_format),
#                 't1_1_motion_mats': Dataset('t1_motion_mats', directory_format),
#                 'ute_motion_mats': Dataset('ute_motion_mats', directory_format),
#                 'fm_motion_mats': Dataset('fm_motion_mats', directory_format),
#                 'epi1_reference': Dataset('reference', nifti_gz_format),
#                 'epi2_reference': Dataset('reference', nifti_gz_format),
#                 'epi3_reference': Dataset('reference', nifti_gz_format),
#                 'epi4_reference': Dataset('reference', nifti_gz_format),
#                 'epi5_reference': Dataset('reference', nifti_gz_format),
#                 'epi6_reference': Dataset('reference', nifti_gz_format),
#                 'epi7_reference': Dataset('reference', nifti_gz_format),
#                 'epi8_reference': Dataset('reference', nifti_gz_format),
#                 'asl_reference': Dataset('reference', nifti_gz_format),
#                 't1_1_reference': Dataset('reference', nifti_gz_format),
#                 'ute_reference': Dataset('reference', nifti_gz_format),
#                 'fm_reference': Dataset('reference', nifti_gz_format)})
#         study.motion_framing_pipeline().run(work_dir=self.work_dir)
#         self.assertDatasetCreated('frame_start_times.txt', study.name)

#     def test_mc_ian(self):
#         study = self.create_study(
#             MotionDetectionStudy, 'mc_detection_study', inputs={
#                 'epi1': Dataset('epi_1_dicom', dicom_format),
#                 'epi2': Dataset('epi_2_dicom', dicom_format),
#                 'epi3': Dataset('epi_3_dicom', dicom_format),
#                 'asl': Dataset('asl_dicom', dicom_format),
#                 't1_1': Dataset('t1_1_dicom', dicom_format),
#                 't2_1': Dataset('t2_1_dicom', dicom_format),
#                 't2_2': Dataset('t2_2_dicom', dicom_format),
#                 't2_3': Dataset('t2_3_dicom', dicom_format),
#                 't2_4': Dataset('t2_4_dicom', dicom_format),
#                 't2_5': Dataset('t2_5_dicom', dicom_format),
#                 'dwi_1_main': Dataset('dwi_1_main_dicom', dicom_format),
#                 'dwi_1_opposite_to_correct': Dataset('dwi_1_opposite_dicom',
#                                                      dicom_format),
#                 'dwi_1_main_ref': Dataset('dwi_1_opposite_dicom',
#                                           dicom_format),
#                 'dwi_1_opposite_ref': Dataset('dwi_1_main_dicom',
#                                               dicom_format),
#                 'ute': Dataset('ute_dicom', dicom_format),
#                 'fm': Dataset('fm_dicom', dicom_format),
#                 'reference': Dataset('reference_dicom', dicom_format),
#                 'epi1_motion_mats': Dataset('epi1_motion_mats', directory_format),
#                 'epi2_motion_mats': Dataset('epi2_motion_mats', directory_format),
#                 'epi3_motion_mats': Dataset('epi3_motion_mats', directory_format),
#                 'asl_motion_mats': Dataset('asl_motion_mats', directory_format),
#                 't1_1_motion_mats': Dataset('t1_1_motion_mats', directory_format),
#                 't2_1_motion_mats': Dataset('t2_1_motion_mats', directory_format),
#                 't2_2_motion_mats': Dataset('t2_2_motion_mats', directory_format),
#                 't2_3_motion_mats': Dataset('t2_3_motion_mats', directory_format),
#                 't2_4_motion_mats': Dataset('t2_4_motion_mats', directory_format),
#                 't2_5_motion_mats': Dataset('t2_5_motion_mats', directory_format),
#                 'dwi_1_main_motion_mats': Dataset('dwi_1_main_motion_mats', directory_format),
#                 'dwi_1_opposite_motion_mats': Dataset('dwi_1_opposite_motion_mats', directory_format),
#                 'ute_motion_mats': Dataset('ute_motion_mats', directory_format),
#                 'fm_motion_mats': Dataset('fm_motion_mats', directory_format),
#                 'epi1_reference': Dataset('reference', nifti_gz_format),
#                 'epi2_reference': Dataset('reference', nifti_gz_format),
#                 'epi3_reference': Dataset('reference', nifti_gz_format),
#                 'asl_reference': Dataset('reference', nifti_gz_format),
#                 't1_1_reference': Dataset('reference', nifti_gz_format),
#                 't2_1_reference': Dataset('reference', nifti_gz_format),
#                 't2_2_reference': Dataset('reference', nifti_gz_format),
#                 't2_3_reference': Dataset('reference', nifti_gz_format),
#                 't2_4_reference': Dataset('reference', nifti_gz_format),
#                 't2_5_reference': Dataset('reference', nifti_gz_format),
#                 'dwi_reference': Dataset('reference', nifti_gz_format),
#                 'ute_reference': Dataset('reference', nifti_gz_format),
#                 'fm_reference': Dataset('reference', nifti_gz_format)})
#         study.motion_framing_pipeline().run(work_dir=self.work_dir)
#         self.assertDatasetCreated('frame_start_times.txt', study.name)
# 
#     def test_plot_md(self):
#         study = self.create_study(
#             MotionDetectionStudy, 'plot_md', inputs={
#                 'frame_start_times': Dataset('frame_start_times', text_format)})
#         study.frame2ref_alignment_pipeline().run(work_dir=self.work_dir)
#         self.assertDatasetCreated('frame2reference_mats', study.name)
#     def test_t2_mc(self):
#         study = self.create_study(
#             CoregisteredT2Study, 't2_reg_study', inputs={
#                 't2': Dataset('pd', nifti_gz_format),
#                 'reference': Dataset('ref_2', nifti_gz_format)})
#         study.t2_motion_mat_pipeline().run(work_dir=self.work_dir)
#         self.assertDatasetCreated('t2_motion_mats', study.name)
    def test_mc_ian(self):
        study = self.create_study(
            MotionDetectionStudy, 'mc_detection_study', inputs={
                'epi1': Dataset('epi_1_dicom', dicom_format),
                't1_1': Dataset('t1_1_dicom', dicom_format),
                't2_1': Dataset('t2_1_dicom', dicom_format),
                't2_2': Dataset('t2_2_dicom', dicom_format),
                't2_3': Dataset('t2_3_dicom', dicom_format),
                't2_4': Dataset('t2_4_dicom', dicom_format),
                't2_5': Dataset('t2_5_dicom', dicom_format),
                'dwi_1_main': Dataset('dwi_1_main_dicom', dicom_format),
                'dwi2ref_1_to_correct': Dataset('dwi2ref_1_dicom',
                                                     dicom_format),
                'dwi2ref_1_opposite_to_correct': Dataset('dwi2ref_1_opposite_dicom',
                                                     dicom_format),
                'dwi_1_main_ref': Dataset('dwi2ref_1_opposite_dicom',
                                          dicom_format),
                'dwi2ref_1_ref': Dataset('dwi2ref_1_opposite_dicom',
                                              dicom_format),
                'dwi2ref_1_opposite_ref': Dataset('dwi2ref_1_dicom',
                                              dicom_format),
                'ute': Dataset('ute_dicom', dicom_format),
                'fm': Dataset('fm_dicom', dicom_format),
                'ref': Dataset('reference_dicom', dicom_format)})
        study.plot_mean_displacement_pipeline().run(work_dir=self.work_dir)
        self.assertDatasetCreated('frame2reference_mats', study.name)
        self.assertDatasetCreated('umaps_align2ref', study.name)
