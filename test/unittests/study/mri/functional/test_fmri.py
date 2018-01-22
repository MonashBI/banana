#!/usr/bin/env python
from nipype import config
from nianalysis.data_formats import text_matrix_format
config.enable_debug_mode()
from nianalysis.dataset import Dataset  # @IgnorePep8
from nianalysis.data_formats import (nifti_gz_format, rdata_format, zip_format, # @IgnorePep8
                                     directory_format, targz_format, par_format) # @IgnorePep8
from nianalysis.study.mri.functional.fmri import FunctionalMRIStudy  # @IgnorePep8
from nianalysis.testing import BaseTestCase  # @IgnorePep8 @Reimport


class TestFMRI(BaseTestCase):

#     def test_feat(self):
#         study = self.create_study(
#             FunctionalMRIStudy, 'feat', input_datasets={
#                 'field_map_mag': Dataset('field_map_mag', nifti_gz_format),
#                 'field_map_phase': Dataset('field_map_phase', nifti_gz_format),
#                 't1': Dataset('mprage', nifti_gz_format),
#                 'rs_fmri': Dataset('rs_fmri', nifti_gz_format),
#                 'rs_fmri_ref': Dataset('rs_fmri_ref', nifti_gz_format)})
#         study.feat_pipeline().run(work_dir=self.work_dir)
#         self.assertDatasetCreated('feat_dir.zip', study.name)
# 
    def test_fix(self):
        study = self.create_study(
            FunctionalMRIStudy, 'fix', input_datasets={
                'fix_dir': Dataset('feat_dir', zip_format),
                'train_data': Dataset('train_data', rdata_format)})
        study.fix_pipeline().run(work_dir=self.work_dir)
        self.assertDatasetCreated('cleaned_file.nii.gz', study.name)

#     def test_optibet(self):
#         study = self.create_study(
#             FunctionalMRIStudy, 'optibet', input_datasets={
#                 't1': Dataset('mprage', nifti_gz_format)})
#         study.optiBET().run(work_dir=self.work_dir, plugin='MultiProc')
#         self.assertDatasetCreated('betted_file.nii.gz', study.name)
#         self.assertDatasetCreated('betted_mask.nii.gz', study.name)

#     def test_ants(self):
#         study = self.create_study(
#             FunctionalMRIStudy, 'optibet', input_datasets={
#                 'rs_fmri': Dataset('rs_fmri', nifti_gz_format),
#                 'betted_file': Dataset('betted_file', nifti_gz_format)})
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
#             FunctionalMRIStudy, 'filtering', input_datasets={
#                 'rs_fmri': Dataset('rs_fmri', nifti_gz_format),
#                 'betted_file': Dataset('betted_file', nifti_gz_format),
#                 'field_map_mag': Dataset('field_map_mag', nifti_gz_format),
#                 'field_map_phase': Dataset('field_map_phase', nifti_gz_format)})
#         study.rsfMRI_filtering().run(work_dir=self.work_dir,
#                                      plugin='MultiProc')
#         self.assertDatasetCreated('filtered_data.nii.gz', study.name)

#     def test_melodic(self):
#         study = self.create_study(
#             FunctionalMRIStudy, 'melodic', input_datasets={
#                 'filtered_data': Dataset('filtered_func_data', nifti_gz_format)})
#         study.MelodicL1().run(work_dir=self.work_dir,
#                               plugin='MultiProc')
#         self.assertDatasetCreated('melodic_ica.zip', study.name)
 
#     def test_fix(self):
#         study = self.create_study(
#             FunctionalMRIStudy, 'fix', input_datasets={
# #                 'field_map_mag': Dataset('field_map_mag', nifti_gz_format),
# #                 'field_map_phase': Dataset('field_map_phase', nifti_gz_format),
#                 'rsfmri_mask': Dataset('rsfmri_mask', nifti_gz_format),
#                 'rs_fmri': Dataset('rs_fmri', nifti_gz_format),
#                 'melodic_ica': Dataset('melodic_ica', zip_format),
#                 'train_data': Dataset('train_data_new', rdata_format),
#                 'hires2example': Dataset('hires2example', text_matrix_format),
#                 'filtered_data': Dataset('filtered_func_data', nifti_gz_format),
#                 'unwarped_file': Dataset('unwarped', nifti_gz_format),
#                 'mc_par': Dataset('prefiltered_func_data_mcf', par_format),
#                 'betted_file': Dataset('betted_file', nifti_gz_format)})
#         study.PrepareFix().run(work_dir=self.work_dir, plugin='Linear')
#         self.assertDatasetCreated('fix_dir.tar.gz', study.name)
#     def test_feat(self):
#         study = self.create_study(
#             FunctionalMRIStudy, 'preprocessing', input_datasets={
#                 'field_map_mag': Dataset('field_map_mag', nifti_gz_format),
#                 'field_map_phase': Dataset('field_map_phase', nifti_gz_format),
#                 't1': Dataset('mprage', nifti_gz_format),
#                 'rs_fmri': Dataset('rs_fmri', nifti_gz_format)})
#         study.ASPREE_preproc().run(work_dir=self.work_dir, plugin='MultiProc')
#         self.assertDatasetCreated('melodic_dir.zip', study.name)
#     def test_apply_trans(self):
#         study = self.create_study(
#             FunctionalMRIStudy, 'apply_smooth', input_datasets={
#                 'cleaned_file': Dataset('filtered_func_data', nifti_gz_format),
#                 'T12MNI_warp': Dataset('T12MNI_1Warp', nifti_gz_format),
#                 'T12MNI_mat': Dataset('T12MNI_0GenericAffine', text_matrix_format),
# #                 'rs_fmri': Dataset('rs_fmri', nifti_gz_format),
# #                 't1': Dataset('mprage', nifti_gz_format),
# #                 'field_map_mag': Dataset('field_map_mag', nifti_gz_format),
# #                 'train_data': Dataset('train_data_new', rdata_format),
#                 'epi2T1_mat': Dataset('epi2T1_0GenericAffine', text_matrix_format)})
#         study.applySmooth().run(work_dir=self.work_dir, plugin='Linear')
#         self.assertDatasetCreated('smoothed_file.nii.gz', study.name)
