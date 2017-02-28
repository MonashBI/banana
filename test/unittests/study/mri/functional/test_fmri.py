#!/usr/bin/env python
from nipype import config
config.enable_debug_mode()
from nianalysis.dataset import Dataset  # @IgnorePep8
from nianalysis.data_formats import (nifti_gz_format, rdata_format, # @IgnorePep8
                                     directory_format, zip_format) # @IgnorePep8
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

    def test_fix(self):
        study = self.create_study(
            FunctionalMRIStudy, 'fix', input_datasets={
                'feat_dir': Dataset('feat_dir', zip_format),
                'train_data': Dataset('train_data', rdata_format)})
        study.fix_pipeline().run(work_dir=self.work_dir)
        self.assertDatasetCreated('cleaned_file.nii.gz', study.name)
