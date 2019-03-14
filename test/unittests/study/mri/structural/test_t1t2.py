from nipype import config
config.enable_debug_mode()
from arcana.data import FilesetInput  # @IgnorePep8
from banana.study.mri.structural.t1t2 import T1T2Study # @IgnorePep8
from banana.file_format import (  # @IgnorePep8
    nifti_gz_format)
from banana.testing import BaseTestCase  # @IgnorePep8
import unittest  # @IgnorePep8


class TestT1T2Study(BaseTestCase):

    def test_t2_registration_pipeline(self):
        study = self.create_study(
            T1T2Study, 't2_registration', inputs=[
                FilesetInput('t1', 'mprage', nifti_gz_format),
                FilesetInput('t2', 'flair', nifti_gz_format)])
        study.t2_registration_pipeline().run(work_dir=self.work_dir)
        self.assertFilesetCreated('t2_coreg.nii.gz', study.name)

#     @unittest.skip("Takes too long for regular tests")
    def test_freesurfer_pipeline(self):
        study = self.create_study(
            T1T2Study, 'freesurfer', inputs=[
                FilesetInput('t1', 'mprage', nifti_gz_format),
                FilesetInput('t2', 'flair', nifti_gz_format)])
        study.freesurfer_pipeline().run(work_dir=self.work_dir)
        self.assertFilesetCreated('fs_recon_all.fs.zip')

    def test_brain_extraction_pipelines(self):
        study = self.create_study(
            T1T2Study, 'brain_mask', inputs=[
                FilesetInput('t1', 'mprage', nifti_gz_format),
                FilesetInput('t2', 'flair', nifti_gz_format),
                FilesetInput('manual_wmh_mask', nifti_gz_format,
                             'manual_wmh_mask')])
        study.t1_brain_extraction_pipeline().run(work_dir=self.work_dir)
        study.manual_wmh_mask_registration_pipeline().run(
            work_dir=self.work_dir)
        for fname in ('t1_brain.nii.gz', 't2_brain.nii.gz',
                      'brain_mask.nii.gz', 'manual_wmh_mask_coreg.nii.gz'):
            self.assertFilesetCreated(fname, study.name)
