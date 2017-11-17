#!/usr/bin/env python
from nipype import config
config.enable_debug_mode()
from nianalysis.dataset import Dataset  # @IgnorePep8
from nianalysis.data_formats import dicom_format  # @IgnorePep8
from nianalysis.study.mri.phantom import QCStudy  # @IgnorePep8
from nianalysis.testing import BaseTestCase as TestCase  # @IgnorePep8 @Reimport


class TestMRI(TestCase):

    def test_phantom_qc(self):
        study = self.create_study(
            QCStudy, 'qc_study', inputs={
                'phantom': Dataset('phantom_t1_09', dicom_format)})
        study.qc_metrics_pipeline().run(work_dir=self.work_dir)
        self.assertDatasetCreated('signal.nii.gz', study.name)
        self.assertDatasetCreated('ghost.nii.gz', study.name)
        self.assertDatasetCreated('background.nii.gz', study.name)
        self.assertField('snr', 1.0, study.name)
        self.assertField('uniformity', 1.0, study.name)
        self.assertField('ghost_intensity', 1.0, study.name)
