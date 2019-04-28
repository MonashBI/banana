#!/usr/bin/env python
from nipype import config
config.enable_debug_mode() # @IgnorePep8
from banana.study.mri.functional.bold import create_multi_fmri_class # @IgnorePep8 @Reimport
from banana.testing import BaseTestCase as TestCase # @IgnorePep8 @Reimport


t1 = 't1'
fm_mag = 'field_map_mag'
fm_phase = 'field_map_phase'
epis = 'epi_1'
train_set = 'rs_train'


class TestFMRI(TestCase):

    def test_fmri(self):

        fMRI, inputs, output_files = create_multi_fmri_class(
            'fMRI', t1, epis, 1, fm_mag=fm_mag, fm_phase=fm_phase,
            training_set=train_set)

        study = self.create_study(
            fMRI, 'fMRI', inputs=inputs,
            enforce_inputs=False)
        study.data(output_files)
        self.assertFilesetCreated(output_files[0]+'.nii.gz', study.name)
