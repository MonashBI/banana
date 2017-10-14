#!/usr/bin/env python
from nipype import config
config.enable_debug_mode()
from nianalysis.dataset import Dataset  # @IgnorePep8
from nianalysis.data_formats import nifti_gz_format, text_matrix_format  # @IgnorePep8
from nianalysis.study.mri.coregistered import (  # @IgnorePep8
    CoregisteredStudy, CoregisteredToMatrixStudy)
from nianalysis.testing import BaseTestCase as TestCase  # @IgnorePep8 @Reimport


class TestCoregistered(TestCase):

    def test_registration(self):
        study = self.create_study(
            CoregisteredStudy, 'registration',
            inputs={
                'to_register': Dataset('flair', nifti_gz_format),
                'reference': Dataset('mprage', nifti_gz_format)})
        pipeline = study.registration_pipeline()
        pipeline.run(work_dir=self.work_dir)
        self.assertDatasetCreated('registered.nii.gz', study.name)
        self.assertDatasetCreated('matrix.mat', study.name)
        # Move the generated matrix file to a location that won't be cleaned

    def test_registration_to_matrix(self):
        study = self.create_study(
            CoregisteredToMatrixStudy, 'registration_to_matrix', {
                'to_register': Dataset('flair', nifti_gz_format),
                'reference': Dataset('mprage', nifti_gz_format),
                'matrix': Dataset('matrix', text_matrix_format)})
        study.registration_pipeline().run(work_dir=self.work_dir)
        self.assertDatasetCreated('registered.nii.gz', study.name)
