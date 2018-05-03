#!/usr/bin/env python
from nipype import config
config.enable_debug_mode()
from nianalysis.dataset import DatasetMatch  # @IgnorePep8
from mbianalysis.data_format import nifti_gz_format  # @IgnorePep8
from mbianalysis.study.mri.base import MRIStudy  # @IgnorePep8
from mbianalysis.testing import BaseTestCase as TestCase  # @IgnorePep8 @Reimport


class TestMRI(TestCase):

    def test_brain_mask(self):
        study = self.create_study(
            MRIStudy, 'mask_study', inputs=[
                DatasetMatch('primary', nifti_gz_format, 'flair'),
                DatasetMatch('coreg_ref', nifti_gz_format, 'mprage')])
        coreg_brain = study.data('coreg_brain')[0]
        self.assertDatasetEqual(coreg_brain,
                                self.reference('coreg_brain'))
