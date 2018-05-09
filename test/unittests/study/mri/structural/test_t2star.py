import logging  # @IgnorePep8
from nipype import config
config.enable_debug_mode()
from arcana.dataset import DatasetMatch  # @IgnorePep8
from mbianalysis.testing import BaseTestCase as TestCase  # @IgnorePep8 @Reimport

from mbianalysis.data_format import zip_format  # @IgnorePep8
from mbianalysis.study.mri.structural.t2star import T2StarStudy  # @IgnorePep8

logger = logging.getLogger('Arcana')


class TestQSM(TestCase):

#    def test_qsm_pipeline(self):
#       study = self.create_study(
#            T2StarStudy, 'qsm', input_datasets={
#                DatasetMatch('coils', zip_format, 'swi_coils')})    
#        study.qsm_pipeline().run(work_dir=self.work_dir)
#        for fname in ('qsm.nii.gz', 'tissue_phase.nii.gz',
#                      'tissue_mask.nii.gz', 'qsm_mask.nii.gz'):
#            self.assertDatasetCreated(dataset_name=fname,
#                                      study_name=study.name)
#            
    def test_qsm_de_pipeline(self):
        study = self.create_study(
            T2StarStudy, 'qsm', inputs=[
                DatasetMatch('coils', zip_format, 'swi_coils')])
        study.qsm_pipeline().run(work_dir=self.work_dir)
        for fname in ('qsm.nii.gz', 'tissue_phase.nii.gz',
                      'tissue_mask.nii.gz', 'qsm_mask.nii.gz'):
            self.assertDatasetCreated(dataset_name=fname,
                                      study_name=study.name)