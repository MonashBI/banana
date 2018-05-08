#!/usr/bin/env python
from nipype import config
from mbianalysis.data_format import rdata_format, dicom_format
config.enable_debug_mode()
from nianalysis.dataset import DatasetMatch  # @IgnorePep8
from mbianalysis.study.mri.functional.fmri_new import FunctionalMRIStudy  # @IgnorePep8
from mbianalysis.testing import BaseTestCase  # @IgnorePep8 @Reimport


class TestFMRI(BaseTestCase):

    def test_fix(self):
        study = self.create_study(
            FunctionalMRIStudy, 'fix', inputs={
                DatasetMatch('t1_primary', dicom_format, 'mprage'),
                DatasetMatch('fm_mag_primary', dicom_format, 'field_map_mag'),
                DatasetMatch('fm_phase_primary', dicom_format,
                             'field_map_phase'),
                DatasetMatch('epi_primary', dicom_format, 'rs_fmri'),
                DatasetMatch('train_data', rdata_format, 'train_data')})
        study.data('melodic_ica')
        self.assertDatasetCreated('melodic_ica.zip', study.name)
