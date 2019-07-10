#!/usr/bin/env python
from nipype import config
config.enable_debug_mode()
from arcana.data import InputFilesets
from banana.file_format import (nifti_gz_format) # @IgnorePep8
from banana.study.pet.static.sPET import StaticPetStudy
from banana.testing import BaseTestCase


class TestsPET(BaseTestCase):

    def test_suvr(self):
        study = self.create_study(
            StaticPetStudy, 'suvr', inputs=[
                InputFilesets('registered_volume', 'suvr_registered_volume', nifti_gz_format),
                InputFilesets('base_mask', 'cerebellum_mask', nifti_gz_format)])
        study.suvr_pipeline().run(work_dir=self.work_dir, plugin='Linear')
        self.assertFilesetCreated('SUVR_image.nii.gz', study.name)
#         self.assertFilesetCreated('t.png', study.name)
