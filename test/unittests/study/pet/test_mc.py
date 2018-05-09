from nipype import config
from nianalysis.study.mri.structural.t1 import CoregisteredT1Study
from nianalysis.study.mri.structural.t2 import CoregisteredT2Study
from nianalysis.data_format import directory_format, text_format
config.enable_debug_mode()
from arcana.dataset import DatasetMatch, Field  # @IgnorePep8
from nianalysis.data_format import nifti_gz_format, list_mode_format  # @IgnorePep8
from nianalysis.study.mri.epi import CoregisteredEPIStudy  # @IgnorePep8
from nianalysis.study.pet.pca_motion_detection import PETPCAMotionDetectionStudy
from nianalysis.testing import BaseTestCase as TestCase  # @IgnorePep8 @Reimport
from nianalysis.study.mri.structural.diffusion_coreg import CoregisteredDiffusionStudy  # @IgnorePep8
from nianalysis.study.mri.motion_detection_ian_2 import MotionDetectionStudy

class TestMC(TestCase):

#     def test_epi_mc(self):
#         study = self.create_study(
#             CoregisteredEPIStudy, 'epi_reg_study', inputs=[
#                 DatasetMatch('epi', nifti_gz_format, 'epi'),
#                 DatasetMatch('reference', nifti_gz_format, 'reference')})
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
#             CoregisteredT1Study, 't1_reg_study', inputs=[
#                 DatasetMatch('t1', nifti_gz_format, 't1'),
#                 DatasetMatch('reference', nifti_gz_format, 'reference')})
#         study.t1_motion_mat_pipeline().run(work_dir=self.work_dir)
#         self.assertDatasetCreated('t1_motion_mats', study.name)

#     def test_hd_info_extraction(self):
#         study = self.create_study(
#             MRIStudy, 't2_reg_study', inputs=[
#                 DatasetMatch('dicom_file', dicom_format, 'dwi')})
#         study.header_info_extraction_pipeline().run(work_dir=self.work_dir)
#         self.assertField('tr', 2.45, study.name)
#         self.assertField('start_time', '124629.127500', study.name)
#         self.assertField('tot_duration', '602', study.name)
#         self.assertField('real_duration', '592.9', study.name)
#         self.assertField('ped', '', study.name)
#         self.assertField('phase_offset', '', study.name)

#     def test_dwi_mc(self):
#         study = self.create_study(
#             MotionDetectionStudy, 'mc_detection_study', inputs=[
#                 DatasetMatch('epi1', dicom_format, 'epi_dicom'),
#                 DatasetMatch('epi2', dicom_format, 'epi2_dicom'),
#                 DatasetMatch('epi3', dicom_format, 'epi3_dicom'),
#                 DatasetMatch('epi4', dicom_format, 'epi4_dicom'),
#                 DatasetMatch('epi5', dicom_format, 'epi5_dicom'),
#                 DatasetMatch('epi6', dicom_format, 'epi6_dicom'),
#                 DatasetMatch('epi7', dicom_format, 'epi7_dicom'),
#                 DatasetMatch('epi8', dicom_format, 'epi8_dicom'),
#                 DatasetMatch('asl', dicom_format, 'asl_dicom'),
#                 DatasetMatch('t1_1', dicom_format, 't1_dicom'),
#                 DatasetMatch('ute', dicom_format, 'ute_dicom'),
#                 DatasetMatch('fm', dicom_format, 'fm_dicom'),
#                 DatasetMatch('reference', dicom_format, 'reference_dicom'),
#                 DatasetMatch('epi1_motion_mats', directory_format, 'epi1_motion_mats'),
#                 DatasetMatch('epi2_motion_mats', directory_format, 'epi2_motion_mats'),
#                 DatasetMatch('epi3_motion_mats', directory_format, 'epi3_motion_mats'),
#                 DatasetMatch('epi4_motion_mats', directory_format, 'epi4_motion_mats'),
#                 DatasetMatch('epi5_motion_mats', directory_format, 'epi5_motion_mats'),
#                 DatasetMatch('epi6_motion_mats', directory_format, 'epi6_motion_mats'),
#                 DatasetMatch('epi7_motion_mats', directory_format, 'epi7_motion_mats'),
#                 DatasetMatch('epi8_motion_mats', directory_format, 'epi8_motion_mats'),
#                 DatasetMatch('asl_motion_mats', directory_format, 'asl_motion_mats'),
#                 DatasetMatch('t1_1_motion_mats', directory_format, 't1_motion_mats'),
#                 DatasetMatch('ute_motion_mats', directory_format, 'ute_motion_mats'),
#                 DatasetMatch('fm_motion_mats', directory_format, 'fm_motion_mats'),
#                 DatasetMatch('epi1_reference', nifti_gz_format, 'reference'),
#                 DatasetMatch('epi2_reference', nifti_gz_format, 'reference'),
#                 DatasetMatch('epi3_reference', nifti_gz_format, 'reference'),
#                 DatasetMatch('epi4_reference', nifti_gz_format, 'reference'),
#                 DatasetMatch('epi5_reference', nifti_gz_format, 'reference'),
#                 DatasetMatch('epi6_reference', nifti_gz_format, 'reference'),
#                 DatasetMatch('epi7_reference', nifti_gz_format, 'reference'),
#                 DatasetMatch('epi8_reference', nifti_gz_format, 'reference'),
#                 DatasetMatch('asl_reference', nifti_gz_format, 'reference'),
#                 DatasetMatch('t1_1_reference', nifti_gz_format, 'reference'),
#                 DatasetMatch('ute_reference', nifti_gz_format, 'reference'),
#                 DatasetMatch('fm_reference', nifti_gz_format, 'reference')})
#         study.motion_framing_pipeline().run(work_dir=self.work_dir)
#         self.assertDatasetCreated('frame_start_times.txt', study.name)

#     def test_mc_ian(self):
#         study = self.create_study(
#             MotionDetectionStudy, 'mc_detection_study', inputs=[
#                 DatasetMatch('epi1', dicom_format, 'epi_1_dicom'),
#                 DatasetMatch('epi2', dicom_format, 'epi_2_dicom'),
#                 DatasetMatch('epi3', dicom_format, 'epi_3_dicom'),
#                 DatasetMatch('asl', dicom_format, 'asl_dicom'),
#                 DatasetMatch('t1_1', dicom_format, 't1_1_dicom'),
#                 DatasetMatch('t2_1', dicom_format, 't2_1_dicom'),
#                 DatasetMatch('t2_2', dicom_format, 't2_2_dicom'),
#                 DatasetMatch('t2_3', dicom_format, 't2_3_dicom'),
#                 DatasetMatch('t2_4', dicom_format, 't2_4_dicom'),
#                 DatasetMatch('t2_5', dicom_format, 't2_5_dicom'),
#                 DatasetMatch('dwi_1_main', dicom_format, 'dwi_1_main_dicom'),
#                 'dwi_1_opposite_to_correct': Dataset('dwi_1_opposite_dicom',
#                                                      dicom_format),
#                 'dwi_1_main_ref': Dataset('dwi_1_opposite_dicom',
#                                           dicom_format),
#                 'dwi_1_opposite_ref': Dataset('dwi_1_main_dicom',
#                                               dicom_format),
#                 DatasetMatch('ute', dicom_format, 'ute_dicom'),
#                 DatasetMatch('fm', dicom_format, 'fm_dicom'),
#                 DatasetMatch('reference', dicom_format, 'reference_dicom'),
#                 DatasetMatch('epi1_motion_mats', directory_format, 'epi1_motion_mats'),
#                 DatasetMatch('epi2_motion_mats', directory_format, 'epi2_motion_mats'),
#                 DatasetMatch('epi3_motion_mats', directory_format, 'epi3_motion_mats'),
#                 DatasetMatch('asl_motion_mats', directory_format, 'asl_motion_mats'),
#                 DatasetMatch('t1_1_motion_mats', directory_format, 't1_1_motion_mats'),
#                 DatasetMatch('t2_1_motion_mats', directory_format, 't2_1_motion_mats'),
#                 DatasetMatch('t2_2_motion_mats', directory_format, 't2_2_motion_mats'),
#                 DatasetMatch('t2_3_motion_mats', directory_format, 't2_3_motion_mats'),
#                 DatasetMatch('t2_4_motion_mats', directory_format, 't2_4_motion_mats'),
#                 DatasetMatch('t2_5_motion_mats', directory_format, 't2_5_motion_mats'),
#                 DatasetMatch('dwi_1_main_motion_mats', directory_format, 'dwi_1_main_motion_mats'),
#                 DatasetMatch('dwi_1_opposite_motion_mats', directory_format, 'dwi_1_opposite_motion_mats'),
#                 DatasetMatch('ute_motion_mats', directory_format, 'ute_motion_mats'),
#                 DatasetMatch('fm_motion_mats', directory_format, 'fm_motion_mats'),
#                 DatasetMatch('epi1_reference', nifti_gz_format, 'reference'),
#                 DatasetMatch('epi2_reference', nifti_gz_format, 'reference'),
#                 DatasetMatch('epi3_reference', nifti_gz_format, 'reference'),
#                 DatasetMatch('asl_reference', nifti_gz_format, 'reference'),
#                 DatasetMatch('t1_1_reference', nifti_gz_format, 'reference'),
#                 DatasetMatch('t2_1_reference', nifti_gz_format, 'reference'),
#                 DatasetMatch('t2_2_reference', nifti_gz_format, 'reference'),
#                 DatasetMatch('t2_3_reference', nifti_gz_format, 'reference'),
#                 DatasetMatch('t2_4_reference', nifti_gz_format, 'reference'),
#                 DatasetMatch('t2_5_reference', nifti_gz_format, 'reference'),
#                 DatasetMatch('dwi_reference', nifti_gz_format, 'reference'),
#                 DatasetMatch('ute_reference', nifti_gz_format, 'reference'),
#                 DatasetMatch('fm_reference', nifti_gz_format, 'reference')})
#         study.motion_framing_pipeline().run(work_dir=self.work_dir)
#         self.assertDatasetCreated('frame_start_times.txt', study.name)
# 
#     def test_plot_md(self):
#         study = self.create_study(
#             MotionDetectionStudy, 'plot_md', inputs=[
#                 DatasetMatch('frame_start_times', text_format, 'frame_start_times')})
#         study.frame2ref_alignment_pipeline().run(work_dir=self.work_dir)
#         self.assertDatasetCreated('frame2reference_mats', study.name)
#     def test_t2_mc(self):
#         study = self.create_study(
#             CoregisteredT2Study, 't2_reg_study', inputs=[
#                 DatasetMatch('t2', nifti_gz_format, 'pd'),
#                 DatasetMatch('reference', nifti_gz_format, 'ref_2')})
#         study.t2_motion_mat_pipeline().run(work_dir=self.work_dir)
#         self.assertDatasetCreated('t2_motion_mats', study.name)
    def test_mc_ian(self):
        study = self.create_study(
            PETPCAMotionDetectionStudy, 'mc_detection_study', inputs=[
                DatasetMatch('list_mode', list_mode_format, 'list_mode')])
#                 'time_offset': 0,
#                 'num_frames': 10,
#                 'temporal_length': 2}])
        study.sinogram_unlisting_pipeline().run(work_dir=self.work_dir)
        self.assertDatasetCreated('ssrb_sinograms', study.name)
