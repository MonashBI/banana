#!/usr/bin/env python
from nipype import config
config.enable_debug_mode()
from arcana.data import FilesetSelector  # @IgnorePep8
from banana.file_format import nifti_gz_format, text_matrix_format  # @IgnorePep8
from banana.study.mri.coregistered import (  # @IgnorePep8
    CoregisteredStudy, CoregisteredToMatrixStudy)
from banana.testing import BaseTestCase as TestCase  # @IgnorePep8 @Reimport


class TestCoregistered(TestCase):

    def test_registration(self):
        study = self.create_study(
            CoregisteredStudy, 'registration',
            inputs=[
                FilesetSelector('to_register', nifti_gz_format, 'flair'),
                FilesetSelector('reference', nifti_gz_format, 'mprage')])
        pipeline = study.linear_registration_pipeline()
        pipeline.run(work_dir=self.work_dir)
        self.assertFilesetCreated('registered.nii.gz', study.name)
        self.assertFilesetCreated('matrix.mat', study.name)
        # Move the generated matrix file to a location that won't be cleaned

    def test_registration_to_matrix(self):
        study = self.create_study(
            CoregisteredToMatrixStudy, 'registration_to_matrix', {
                FilesetSelector('to_register', nifti_gz_format, 'flair'),
                FilesetSelector('reference', nifti_gz_format, 'mprage'),
                FilesetSelector('matrix', text_matrix_format, 'matrix')})
        study.linear_registration_pipeline().run(work_dir=self.work_dir)
        self.assertFilesetCreated('registered.nii.gz', study.name)
