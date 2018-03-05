from nipype import config
config.enable_debug_mode()
from nianalysis.dataset import Dataset  # @IgnorePep8
from mbianalysis.study.mri.structural.t1t2 import T1T2Study # @IgnorePep8
from nianalysis.data_formats import (  # @IgnorePep8
    nifti_gz_format)
from mbianalysis.testing import BaseTestCase  # @IgnorePep8
import unittest  # @IgnorePep8


class TestT1T2Study(BaseTestCase):

    def test_t2_registration_pipeline(self):
        study = self.create_study(
            T1T2Study, 't2_registration', inputs={
                't1': Dataset('mprage', nifti_gz_format),
                't2': Dataset('flair', nifti_gz_format)})
        study.t2_registration_pipeline().run(work_dir=self.work_dir)
        self.assertDatasetCreated('t2_coreg.nii.gz', study.name)

#     @unittest.skip("Takes too long for regular tests")
    def test_freesurfer_pipeline(self):
        study = self.create_study(
            T1T2Study, 'freesurfer', inputs={
                't1': Dataset('mprage', nifti_gz_format),
                't2': Dataset('flair', nifti_gz_format)})
        study.freesurfer_pipeline().run(work_dir=self.work_dir)
        self.assertDatasetCreated('fs_recon_all.fs.zip')

    def test_brain_mask_pipelines(self):
        study = self.create_study(
            T1T2Study, 'brain_mask', inputs={
                't1': Dataset('mprage', nifti_gz_format),
                't2': Dataset('flair', nifti_gz_format),
                'manual_wmh_mask': Dataset('manual_wmh_mask',
                                           nifti_gz_format)})
        study.t1_brain_mask_pipeline().run(work_dir=self.work_dir)
        study.manual_wmh_mask_registration_pipeline().run(
            work_dir=self.work_dir)
        for fname in ('t1_masked.nii.gz', 't2_masked.nii.gz',
                      'brain_mask.nii.gz', 'manual_wmh_mask_coreg.nii.gz'):
            self.assertDatasetCreated(fname, study.name)
