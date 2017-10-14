#!/usr/bin/env python
from nipype import config
config.enable_debug_mode()
from nianalysis.dataset import Dataset  # @IgnorePep8
from nianalysis.data_formats import nifti_gz_format  # @IgnorePep8
from nianalysis.study.mri.base import MRIStudy  # @IgnorePep8
from nianalysis.testing import BaseTestCase as TestCase  # @IgnorePep8 @Reimport


class TestMRI(TestCase):

    def test_brain_mask(self):
        study = self.create_study(
            MRIStudy, 'mask_study', inputs={
                'primary': Dataset('flair', nifti_gz_format)})
        study.brain_mask_pipeline().run(work_dir=self.work_dir)
        self.assertDatasetCreated('brain_mask.nii.gz', study.name)
