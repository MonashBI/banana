#!/usr/bin/env python
from nipype import config
config.enable_debug_mode()
from nianalysis.dataset import Dataset  # @IgnorePep8
from nianalysis.data_format import (nifti_gz_format) # @IgnorePep8
from mbianalysis.study.pet.dynamic.dPET import DynamicPETStudy  # @IgnorePep8
from mbianalysis.testing import BaseTestCase  # @IgnorePep8 @Reimport


class TestdPET(BaseTestCase):

    def test_reg(self):
        study = self.create_study(
            DynamicPETStudy, 'reg', inputs={
                'pet_volumes': Dataset('pet_image', nifti_gz_format)})
        study.ICA_pipeline().run(work_dir=self.work_dir,
                                             plugin='Linear')
        self.assertDatasetCreated('decomposed_file.nii.gz', study.name)
#         self.assertDatasetCreated('t.png', study.name)
