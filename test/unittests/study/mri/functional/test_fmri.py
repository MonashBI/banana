#!/usr/bin/env python
from nipype import config
from mbianalysis.data_format import text_matrix_format
config.enable_debug_mode()
from nianalysis.dataset import DatasetMatch  # @IgnorePep8
from mbianalysis.data_format import (nifti_gz_format, rdata_format, zip_format, # @IgnorePep8
                                     directory_format, targz_format, par_format) # @IgnorePep8
from mbianalysis.study.mri.functional.fmri import FunctionalMRIStudy  # @IgnorePep8
from mbianalysis.testing import BaseTestCase  # @IgnorePep8 @Reimport


class TestFMRI(BaseTestCase):

#     def test_feat(self):
#         study = self.create_study(
#             FunctionalMRIStudy, 'feat', inputs=[
#                 DatasetMatch('field_map_mag', nifti_gz_format, 'field_map_mag'),
#                 DatasetMatch('field_map_phase', nifti_gz_format, 'field_map_phase'),
#                 DatasetMatch('t1', nifti_gz_format, 'mprage'),
#                 DatasetMatch('rs_fmri', nifti_gz_format, 'rs_fmri'),
#                 DatasetMatch('rs_fmri_ref', nifti_gz_format, 'rs_fmri_ref')})
#         study.feat_pipeline().run(work_dir=self.work_dir)
#         self.assertDatasetCreated('feat_dir.zip', study.name)
# 
    def test_fix(self):
        study = self.create_study(
            FunctionalMRIStudy, 'fix', input_datasets={
                DatasetMatch('fix_dir', zip_format, 'feat_dir'),
                DatasetMatch('train_data', rdata_format, 'train_data')})
        study.fix_pipeline().run(work_dir=self.work_dir)
        self.assertDatasetCreated('cleaned_file.nii.gz', study.name)

#     def test_optibet(self):
#         study = self.create_study(
#             FunctionalMRIStudy, 'optibet', inputs=[
#                 DatasetMatch('t1', nifti_gz_format, 'mprage')})
#         study.optiBET().run(work_dir=self.work_dir, plugin='MultiProc')
#         self.assertDatasetCreated('betted_file.nii.gz', study.name)
#         self.assertDatasetCreated('betted_mask.nii.gz', study.name)

#     def test_ants(self):
#         study = self.create_study(
#             FunctionalMRIStudy, 'optibet', inputs=[
#                 DatasetMatch('rs_fmri', nifti_gz_format, 'rs_fmri'),
#                 DatasetMatch('betted_file', nifti_gz_format, 'betted_file')})
#         study.ANTsRegistration().run(work_dir=self.work_dir,
#                                      plugin='MultiProc')
#         self.assertDatasetCreated('epi2T1.nii.gz', study.name)
#         self.assertDatasetCreated('epi2T1_mat.mat', study.name)
#         self.assertDatasetCreated('T12MNI_linear.nii.gz', study.name)
#         self.assertDatasetCreated('T12MNI_mat.mat', study.name)
#         self.assertDatasetCreated('T12MNI_warp.nii.gz', study.name)
#         self.assertDatasetCreated('T12MNI_invwarp.nii.gz', study.name)

#     def test_filtering(self):
#         study = self.create_study(
#             FunctionalMRIStudy, 'filtering', inputs=[
#                 DatasetMatch('rs_fmri', nifti_gz_format, 'rs_fmri'),
#                 DatasetMatch('betted_file', nifti_gz_format, 'betted_file'),
#                 DatasetMatch('field_map_mag', nifti_gz_format, 'field_map_mag'),
#                 DatasetMatch('field_map_phase', nifti_gz_format, 'field_map_phase')})
#         study.rsfMRI_filtering().run(work_dir=self.work_dir,
#                                      plugin='MultiProc')
#         self.assertDatasetCreated('filtered_data.nii.gz', study.name)

#     def test_melodic(self):
#         study = self.create_study(
#             FunctionalMRIStudy, 'melodic', inputs=[
#                 DatasetMatch('filtered_data', nifti_gz_format, 'filtered_func_data')})
#         study.MelodicL1().run(work_dir=self.work_dir,
#                               plugin='MultiProc')
#         self.assertDatasetCreated('melodic_ica.zip', study.name)
 
#     def test_fix(self):
#         study = self.create_study(
#             FunctionalMRIStudy, 'fix', input_datasets={
# #                 DatasetMatch('field_map_mag', nifti_gz_format, 'field_map_mag'),
# #                 DatasetMatch('field_map_phase', nifti_gz_format, 'field_map_phase'),
#                 DatasetMatch('rsfmri_mask', nifti_gz_format, 'rsfmri_mask'),
#                 DatasetMatch('rs_fmri', nifti_gz_format, 'rs_fmri'),
#                 DatasetMatch('melodic_ica', zip_format, 'melodic_ica'),
#                 DatasetMatch('train_data', rdata_format, 'train_data_new'),
#                 DatasetMatch('hires2example', text_matrix_format, 'hires2example'),
#                 DatasetMatch('filtered_data', nifti_gz_format, 'filtered_func_data'),
#                 DatasetMatch('unwarped_file', nifti_gz_format, 'unwarped'),
#                 DatasetMatch('mc_par', par_format, 'prefiltered_func_data_mcf'),
#                 DatasetMatch('betted_file', nifti_gz_format, 'betted_file')})
#         study.PrepareFix().run(work_dir=self.work_dir, plugin='Linear')
#         self.assertDatasetCreated('fix_dir.tar.gz', study.name)
#     def test_feat(self):
#         study = self.create_study(
#             FunctionalMRIStudy, 'preprocessing', inputs=[
#                 DatasetMatch('field_map_mag', nifti_gz_format, 'field_map_mag'),
#                 DatasetMatch('field_map_phase', nifti_gz_format, 'field_map_phase'),
#                 DatasetMatch('t1', nifti_gz_format, 'mprage'),
#                 DatasetMatch('rs_fmri', nifti_gz_format, 'rs_fmri')})
#         study.ASPREE_preproc().run(work_dir=self.work_dir, plugin='MultiProc')
#         self.assertDatasetCreated('melodic_dir.zip', study.name)
#     def test_apply_trans(self):
#         study = self.create_study(
#             FunctionalMRIStudy, 'apply_smooth', inputs=[
#                 DatasetMatch('cleaned_file', nifti_gz_format, 'filtered_func_data'),
#                 DatasetMatch('T12MNI_warp', nifti_gz_format, 'T12MNI_1Warp'),
#                 DatasetMatch('T12MNI_mat', text_matrix_format, 'T12MNI_0GenericAffine'),
# #                 DatasetMatch('rs_fmri', nifti_gz_format, 'rs_fmri'),
# #                 DatasetMatch('t1', nifti_gz_format, 'mprage'),
# #                 DatasetMatch('field_map_mag', nifti_gz_format, 'field_map_mag'),
# #                 DatasetMatch('train_data', rdata_format, 'train_data_new'),
#                 DatasetMatch('epi2T1_mat', text_matrix_format, 'epi2T1_0GenericAffine')})
#         study.applySmooth().run(work_dir=self.work_dir, plugin='Linear')
#         self.assertDatasetCreated('smoothed_file.nii.gz', study.name)
