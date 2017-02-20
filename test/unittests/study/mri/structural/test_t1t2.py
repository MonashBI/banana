#!/usr/bin/env python

import os.path  # @IgnorePep8
from nipype import config
config.enable_debug_mode()
from nianalysis.dataset import Dataset  # @IgnorePep8
from nianalysis.study.mri.structural.t1t2 import T1T2Study # @IgnorePep8
from nianalysis.archive.local import LocalArchive  # @IgnorePep8
from nianalysis.data_formats import (  # @IgnorePep8
    nifti_format, nifti_gz_format)
if __name__ == '__main__':
    from nianalysis.testing import DummyTestCase as TestCase  # @IgnorePep8 @UnusedImport
else:
    from nianalysis.testing import PipelineTeseCase as TestCase  # @IgnorePep8 @Reimport


class TestT1T2Study(TestCase):

    TEST_MODULE = 'STUDYMRISTRUCTURAL'
    TEST_NAME = 'T1T2'
    REQUIRED_DATASETS = ['t1.nii', 't2.nii', 'mprage.nii.gz', 'flair.nii.gz',
                         'manual_wmh_mask.nii.gz']

    def test_t2_registration_pipeline(self):
        study = self.create_study(
            T1T2Study, 't2_registration', input_datasets={
                't1': Dataset('t1', nifti_format),
                't2': Dataset('t2', nifti_format)})
        study.t2_registration_pipeline().run(work_dir=self.work_dir)
        self.assertDatasetCreated('t2_coreg_t1.nii')

#     def test_freesurfer_pipeline(self):
#         self._remove_generated_files(self.PROJECT_NAME)
#         study = T1T2Study(
#             name=self.DATASET_NAME,
#             project_id=self.PROJECT_NAME,
#             archive=LocalArchive(self.ARCHIVE_PATH),
#             input_datasets={
#                 't1': Dataset('t1', nifti_format),
#                 't2': Dataset('t2', nifti_format)})
#         study.freesurfer_pipeline().run(work_dir=self.WORK_DIR)
#         for fname in ('fs_recon_all.fs.zip',):
#             output_path = os.path.join(
#                 self._session_dir(self.PROJECT_NAME), fname)
#             self.assertTrue(
#                 os.path.exists(output_path),
#                 "Output path '{}' was not created".format(output_path))

    def test_brain_mask_pipelines(self):
        self._remove_generated_files(self.PROJECT_NAME)
        study = T1T2Study(
            name=self.DATASET_NAME,
            project_id=self.PROJECT_NAME,
            archive=LocalArchive(self.ARCHIVE_PATH),
            input_datasets={
                't1': Dataset('mprage', nifti_gz_format),
                't2': Dataset('flair', nifti_gz_format),
                'manual_wmh_mask': Dataset('manual_wmh_mask',
                                           nifti_gz_format)})
        study.t1_brain_mask_pipeline().run(work_dir=self.WORK_DIR)
        study.manual_wmh_mask_registration_pipeline().run(
            work_dir=self.WORK_DIR)
        for fname in ('t1_masked.nii.gz', 't2_masked.nii.gz',
                      'brain_mask.nii.gz', 'manual_wmh_mask_coreg.nii.gz'):
            output_path = os.path.join(
                self._session_dir(self.PROJECT_NAME), fname)
            self.assertTrue(
                os.path.exists(output_path),
                "Output path '{}' was not created".format(output_path))

if __name__ == '__main__':
    tester = TestT1T2Study()
    tester.test_freesurfer_pipeline()
