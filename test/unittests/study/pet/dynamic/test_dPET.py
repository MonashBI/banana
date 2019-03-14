#!/usr/bin/env python
from nipype import config
config.enable_debug_mode()
from arcana.data import FilesetInput  # @IgnorePep8
from banana.file_format import (nifti_gz_format) # @IgnorePep8
from banana.study.pet.dynamic.dPET import DynamicPetStudy  # @IgnorePep8
from banana.testing import BaseTestCase  # @IgnorePep8 @Reimport


class TestdPET(BaseTestCase):

    def test_reg(self):
        study = self.create_study(
            DynamicPetStudy, 'reg', inputs=[
                FilesetInput('pet_volumes', 'pet_image', nifti_gz_format)])
        study.ICA_pipeline().run(work_dir=self.work_dir,
                                             plugin='Linear')
        self.assertFilesetCreated('decomposed_file.nii.gz', study.name)
#         self.assertFilesetCreated('t.png', study.name)
