#!/usr/bin/env python
from nipype import config
config.enable_debug_mode()
from arcana.data import FilesetSelector  # @IgnorePep8
from banana.file_format import (nifti_gz_format) # @IgnorePep8
from banana.study.pet.static.sPET import StaticPETStudy  # @IgnorePep8
from banana.testing import BaseTestCase  # @IgnorePep8 @Reimport


class TestsPET(BaseTestCase):

    def test_suvr(self):
        study = self.create_study(
            StaticPETStudy, 'suvr', inputs=[
                FilesetSelector('registered_volume', 'suvr_registered_volume', nifti_gz_format),
                FilesetSelector('base_mask', 'cerebellum_mask', nifti_gz_format)])
        study.suvr_pipeline().run(work_dir=self.work_dir, plugin='Linear')
        self.assertFilesetCreated('SUVR_image.nii.gz', study.name)
#         self.assertFilesetCreated('t.png', study.name)
