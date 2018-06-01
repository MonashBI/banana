#!/usr/bin/env python
from nipype import config
config.enable_debug_mode() # @IgnorePep8
from nianalysis.study.mri.functional.fmri_new import create_fmri_study_class # @IgnorePep8 @Reimport
from nianalysis.testing import BaseTestCase as TestCase # @IgnorePep8 @Reimport


t1 = 't1'
fm_mag = 'field_map_mag'
fm_phase = 'field_map_phase'
epis = ['epi_1']


class TestFMRI(TestCase):

    def test_fmri(self):

        fMRI, inputs, output_file = create_fmri_study_class(
            'fMRI', t1, epis, fm_mag=fm_mag, fm_phase=fm_phase)
        study = self.create_study(
            fMRI, 'fMRI', inputs=inputs,
            enforce_inputs=False)
        study.data('epi_0_'+output_file)
        self.assertDatasetCreated('epi_0_'+output_file+'.nii.gz', study.name)
