#!/usr/bin/env python
from nipype import config
config.enable_debug_mode()
from nianalysis.dataset import Dataset  # @IgnorePep8
from nianalysis.data_formats import (nifti_gz_format) # @IgnorePep8
from nianalysis.study.pet.dynamic.dPET import DynamicPETStudy  # @IgnorePep8
from nianalysis.testing import BaseTestCase  # @IgnorePep8 @Reimport


class TestdPET(BaseTestCase):

    def test_reg(self):
        study = self.create_study(
            DynamicPETStudy, 'reg', input_datasets={
                'registered_volumes': Dataset('reg_registered_volumes', nifti_gz_format)})
        study.ICA_pipeline().run(work_dir=self.work_dir,
                                             plugin='Linear')
        self.assertDatasetCreated('decomposed_file.nii.gz', study.name)
#         self.assertDatasetCreated('t.png', study.name)
