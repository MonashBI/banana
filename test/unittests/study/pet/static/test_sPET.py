#!/usr/bin/env python
from nipype import config
config.enable_debug_mode()
from nianalysis.dataset import Dataset  # @IgnorePep8
from nianalysis.data_formats import (nifti_gz_format) # @IgnorePep8
from nianalysis.study.pet.static.sPET import StaticPETStudy  # @IgnorePep8
from nianalysis.testing import BaseTestCase  # @IgnorePep8 @Reimport


class TestsPET(BaseTestCase):

    def test_suvr(self):
        study = self.create_study(
            StaticPETStudy, 'suvr', input_datasets={
                'registered_volume': Dataset('suvr_registered_volume', nifti_gz_format),
                'base_mask': Dataset('cerebellum_mask', nifti_gz_format)})
        study.suvr_pipeline().run(work_dir=self.work_dir, plugin='Linear')
        self.assertDatasetCreated('SUVR_image.nii.gz', study.name)
#         self.assertDatasetCreated('t.png', study.name)
