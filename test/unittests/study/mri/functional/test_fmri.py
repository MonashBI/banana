#!/usr/bin/env python
from nipype import config
config.enable_debug_mode() # @IgnorePep8
from nianalysis.study.mri.functional.fmri_new import create_fmri_study_class # @IgnorePep8 @Reimport
from nianalysis.testing import BaseTestCase as TestCase # @IgnorePep8 @Reimport


t1 = 't1'
fm_mag = 'field_map_mag'
fm_phase = 'field_map_phase'
epis = 'epi_1'
train_set = 'rs_train'


class TestFMRI(TestCase):

    def test_fmri(self):

        fMRI, inputs, output_files = create_fmri_study_class(
            'fMRI', t1, epis, 1, fm_mag=fm_mag, fm_phase=fm_phase,
            training_set=train_set)

        study = self.create_study(
            fMRI, 'fMRI', inputs=inputs,
            enforce_inputs=False)
        epi1, epi2 = study.data(output_files)
        self.assertDatasetsEqual(epi1, epi2)
        self.assertDatasetCreated('.nii.gz', study.name)
