from nipype import config
config.enable_debug_mode()
from arcana.data import FilesetSelector  # @IgnorePep8
from nianalysis.study.mri.structural.t1t2 import T1T2Study # @IgnorePep8
from nianalysis.file_format import (  # @IgnorePep8
    nifti_gz_format)
from nianalysis.testing import BaseTestCase  # @IgnorePep8
import unittest  # @IgnorePep8


class TestT1T2Study(BaseTestCase):

    def test_t2_registration_pipeline(self):
        study = self.create_study(
            T1T2Study, 't2_registration', inputs=[
                FilesetSelector('t1', nifti_gz_format, 'mprage'),
                FilesetSelector('t2', nifti_gz_format, 'flair')])
        study.t2_registration_pipeline().run(work_dir=self.work_dir)
        self.assertFilesetCreated('t2_coreg.nii.gz', study.name)

#     @unittest.skip("Takes too long for regular tests")
    def test_freesurfer_pipeline(self):
        study = self.create_study(
            T1T2Study, 'freesurfer', inputs=[
                FilesetSelector('t1', nifti_gz_format, 'mprage'),
                FilesetSelector('t2', nifti_gz_format, 'flair')])
        study.freesurfer_pipeline().run(work_dir=self.work_dir)
        self.assertFilesetCreated('fs_recon_all.fs.zip')

    def test_brain_extraction_pipelines(self):
        study = self.create_study(
            T1T2Study, 'brain_mask', inputs=[
                FilesetSelector('t1', nifti_gz_format, 'mprage'),
                FilesetSelector('t2', nifti_gz_format, 'flair'),
                FilesetSelector('manual_wmh_mask', nifti_gz_format,
                             'manual_wmh_mask')])
        study.t1_brain_extraction_pipeline().run(work_dir=self.work_dir)
        study.manual_wmh_mask_registration_pipeline().run(
            work_dir=self.work_dir)
        for fname in ('t1_brain.nii.gz', 't2_brain.nii.gz',
                      'brain_mask.nii.gz', 'manual_wmh_mask_coreg.nii.gz'):
            self.assertFilesetCreated(fname, study.name)
