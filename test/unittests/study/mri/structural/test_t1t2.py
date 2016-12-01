#!/usr/bin/env python

import os.path  # @IgnorePep8
import shutil
from nipype import config
config.enable_debug_mode()
from nianalysis.dataset import Dataset  # @IgnorePep8
from nianalysis.study.mri.structural.coregistered import T1T2Study # @IgnorePep8
from nianalysis.archive.local import LocalArchive  # @IgnorePep8
from nianalysis.data_formats import (  # @IgnorePep8
    nifti_format)
if __name__ == '__main__':
    from nianalysis.testing import DummyTestCase as TestCase  # @IgnorePep8 @UnusedImport
else:
    from nianalysis.testing import BaseImageTestCase as TestCase  # @IgnorePep8 @Reimport
from nianalysis.testing import test_data_dir  # @IgnorePep8


class TestT1T2Study(TestCase):

    DATASET_NAME = 'T1AndT2'
    PROJECT_NAME = 'T1AndT2'
    ARCHIVE_PATH = os.path.join(test_data_dir, 'archives', 't1_and_t2')

    def setUp(self):
        shutil.rmtree(self.ARCHIVE_PATH, ignore_errors=True)
        if not os.path.exists(self._session_dir(self.PROJECT_NAME)):
            os.makedirs(self._session_dir(self.PROJECT_NAME))
            for fname in ('t1.nii', 't2.nii'):
                shutil.copy(os.path.join(test_data_dir, fname),
                            os.path.join(self._session_dir(self.PROJECT_NAME),
                                         fname))

    def test_coregistration_pipeline(self):
        self._remove_generated_files(self.PROJECT_NAME)
        study = T1T2Study(
            name=self.DATASET_NAME,
            project_id=self.PROJECT_NAME,
            archive=LocalArchive(self.ARCHIVE_PATH),
            input_datasets={
                't1': Dataset('t1', nifti_format),
                't2': Dataset('t2', nifti_format)})
        study.coregistration_pipeline().run()
        output_path = os.path.join(self._session_dir(self.PROJECT_NAME),
                                   't2_coreg_t1.nii'
                                   .format(self.DATASET_NAME))
        self.assert_(os.path.exists(output_path),
                     "Output path '{}' was not created".format(output_path))

    def test_segmentation_pipeline(self):
        self._remove_generated_files(self.PROJECT_NAME)
        study = T1T2Study(
            name=self.DATASET_NAME,
            project_id=self.PROJECT_NAME,
            archive=LocalArchive(self.ARCHIVE_PATH),
            input_datasets={
                't1': Dataset('t1', nifti_format),
                't2': Dataset('t2', nifti_format)})
        study.segmentation_pipeline().run()
        for fname in ('t1_grey_matter.nii', 't1_white_matter.nii',
                      't1_csf.nii'):
            output_path = os.path.join(
                self._session_dir(self.PROJECT_NAME), fname)
            self.assertTrue(
                os.path.exists(output_path),
                "Output path '{}' was not created".format(output_path))

    def test_freesurfer_pipeline(self):
        self._remove_generated_files(self.PROJECT_NAME)
        study = T1T2Study(
            name=self.DATASET_NAME,
            project_id=self.PROJECT_NAME,
            archive=LocalArchive(self.ARCHIVE_PATH),
            input_datasets={
                't1': Dataset('t1', nifti_format),
                't2': Dataset('t2', nifti_format)})
        study.freesurfer_pipeline().run()
        for fname in ('freesurfer',):
            output_path = os.path.join(
                self._session_dir(self.PROJECT_NAME), fname)
            self.assertTrue(
                os.path.exists(output_path),
                "Output path '{}' was not created".format(output_path))

if __name__ == '__main__':
    tester = TestT1T2Study()
    tester.test_freesurfer_pipeline()
