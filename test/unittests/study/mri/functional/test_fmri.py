#!/usr/bin/env python
from nipype import config
config.enable_debug_mode() # @IgnorePep8
from nianalysis.study.multimodal.test_rsfmri import (  # @IgnorePep8 @Reimport
    fMRI, inputs)  # @IgnorePep8
from nianalysis.testing import BaseTestCase as TestCase # @IgnorePep8 @Reimport


class TestFMRI(TestCase):

    def test_fmri(self):

        study = self.create_study(
            fMRI, 'fMRI', inputs=inputs,
            enforce_inputs=False)
        study.data('epi_0_melodic_ica')
        self.assertDatasetCreated('epi_0_melodic_ica.zip', study.name)
