#!/usr/bin/env python
from nipype import config
config.enable_debug_mode()
from nianalysis.dataset import Dataset  # @IgnorePep8
from nianalysis.study.mri.structural.ute import UTEStudy
from nianalysis.data_formats import (  # @IgnorePep8
    dicom_format)
from nianalysis.testing import BaseTestCase as TestCase  # @IgnorePep8 @Reimport


class TestUTE(TestCase):

    def test_ute(self):
        study = self.create_study(
            UTEStudy, 'registration', {
                'ute_echo1': Dataset('ute_echo1', dicom_format),
                'ute_echo2': Dataset('ute_echo2', dicom_format),
                'umap_ute': Dataset('umap_ute', dicom_format)})
        study.registration_pipeline().run(work_dir=self.work_dir)
        self.assertDatasetCreated('ute1_registered.nii.gz', study.name)
        self.assertDatasetCreated('ute2_registered.nii.gz', study.name)
