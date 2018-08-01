#!/usr/bin/env python
from nipype import config
config.enable_debug_mode()
from arcana.data import FilesetMatch  # @IgnorePep8
from nianalysis.file_format import (nifti_gz_format) # @IgnorePep8
from nianalysis.study.pet.dynamic.dPET import DynamicPETStudy  # @IgnorePep8
from nianalysis.testing import BaseTestCase  # @IgnorePep8 @Reimport


class TestdPET(BaseTestCase):

    def test_reg(self):
        study = self.create_study(
            DynamicPETStudy, 'reg', inputs=[
                FilesetMatch('pet_volumes', nifti_gz_format, 'pet_image')])
        study.ICA_pipeline().run(work_dir=self.work_dir,
                                             plugin='Linear')
        self.assertFilesetCreated('decomposed_file.nii.gz', study.name)
#         self.assertFilesetCreated('t.png', study.name)
