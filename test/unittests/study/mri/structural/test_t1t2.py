#!/usr/bin/env python

from nipype import config
config.enable_debug_mode()
from nianalysis.dataset import Dataset  # @IgnorePep8
from nianalysis.study.mri.structural.t1t2 import T1T2Study # @IgnorePep8
from nianalysis.data_formats import (  # @IgnorePep8
    nifti_format, nifti_gz_format)
if __name__ == '__main__':
    from nianalysis.testing import DummyTestCase as TestCase  # @IgnorePep8 @UnusedImport
else:
    from nianalysis.testing import PipelineTeseCase as TestCase  # @IgnorePep8 @Reimport


class TestT1T2Study(TestCase):

    def test_t2_registration_pipeline(self):
        study = self.create_study(
            T1T2Study, 't2_registration', input_datasets={
                't1': Dataset('t1', nifti_format),
                't2': Dataset('t2', nifti_format)})
        study.t2_registration_pipeline().run(work_dir=self.work_dir)
        self.assertDatasetCreated('t2_coreg_t1.nii')

    def test_freesurfer_pipeline(self):
        study = self.create_study(
            T1T2Study, 'freesurfer', input_datasets={
                't1': Dataset('t1', nifti_format),
                't2': Dataset('t2', nifti_format)})
        study.freesurfer_pipeline().run(work_dir=self.work_dir)
        self.assertDatasetCreated('fs_recon_all.fs.zip')

    def test_brain_mask_pipelines(self):
        study = self.create_study(
            T1T2Study, 'brain_mask', input_datasets={
                't1': Dataset('mprage', nifti_gz_format),
                't2': Dataset('flair', nifti_gz_format),
                'manual_wmh_mask': Dataset('manual_wmh_mask',
                                           nifti_gz_format)})
        study.t1_brain_mask_pipeline().run(work_dir=self.work_dir)
        study.manual_wmh_mask_registration_pipeline().run(
            work_dir=self.work_dir)
        for fname in ('t1_masked.nii.gz', 't2_masked.nii.gz',
                      'brain_mask.nii.gz', 'manual_wmh_mask_coreg.nii.gz'):
            self.assertDatasetCreated(fname)

if __name__ == '__main__':
    tester = TestT1T2Study()
    tester.test_freesurfer_pipeline()
