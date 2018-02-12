#!/usr/bin/env python
from nipype import config
config.enable_debug_mode()
from nianalysis.dataset import Dataset  # @IgnorePep8
from nianalysis.data_formats import dicom_format  # @IgnorePep8
from nianalysis.study.mri.phantom import PhantomStudy  # @IgnorePep8
from nianalysis.testing import BaseTestCase as TestCase  # @IgnorePep8 @Reimport


class TestPhantomQC(TestCase):

    def test_phantom_qc(self):
        study = self.create_study(
            PhantomStudy, 'qc_study', inputs={
                't1_32ch_saline': Dataset('phantom_t1_09', dicom_format)})
        study.t1_32ch_qc_metrics_pipeline().run(work_dir=self.work_dir)
        self.assertDatasetCreated('t1_32ch_signal.nii.gz', study.name)
        self.assertDatasetCreated('t1_32ch_ghost.nii.gz', study.name)
        self.assertDatasetCreated('t1_32ch_background.nii.gz', study.name)
        self.assertField('t1_32ch_snr', 2163.19664, study.name,
                         to_places=5)
        self.assertField('t1_32ch_uniformity', 35.93246,
                         study.name, to_places=5)
        self.assertField('t1_32ch_ghost_intensity', 28942.90170,
                         study.name, to_places=5)
