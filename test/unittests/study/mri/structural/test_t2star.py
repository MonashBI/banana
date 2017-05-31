import logging  # @IgnorePep8
from nipype import config
config.enable_debug_mode()
from nianalysis.dataset import Dataset  # @IgnorePep8
from nianalysis.testing import BaseTestCase as TestCase  # @IgnorePep8 @Reimport

from nianalysis.data_formats import zip_format  # @IgnorePep8
from nianalysis.study.mri.structural.t2star import T2StarStudy  # @IgnorePep8

logger = logging.getLogger('NiAnalysis')


class TestQSM(TestCase):

#    def test_qsm_pipeline(self):
#        study = self.create_study(
#           T2StarStudy, 'qsm', input_datasets={
#                'coils': Dataset('swi_coils', zip_format)})
#        study.qsm_pipeline().run(work_dir=self.work_dir)
#        for fname in ('qsm.nii.gz', 'tissue_phase.nii.gz',
#                      'tissue_mask.nii.gz', 'qsm_mask.nii.gz'):
#            self.assertDatasetCreated(dataset_name=fname,
#                                      study_name=study.name)
#            
            
    def test_qsm_de_pipeline(self):
        study = self.create_study(
            T2StarStudy, 'qsm', input_datasets={
<<<<<<< HEAD
                'coils': Dataset('swi_coils_de', zip_format)})
        study.qsm_de_pipeline().run(work_dir=self.work_dir)
        for fname in ('qsm.nii.gz', 'tissue_phase.nii.gz',
                      'tissue_mask.nii.gz', 'qsm_mask.nii.gz'):
            self.assertDatasetCreated(dataset_name=fname,
=======
                'coils': Dataset('swi_coils', zip_format)})
        study.qsm_pipeline().run(work_dir=self.work_dir)
        for fname in ('qsm.nii.gz', 'tissue_phase.nii.gz',
                      'tissue_mask.nii.gz', 'qsm_mask.nii.gz'):
            self.assertDatasetCreated(dataset_name=fname,
                                      study_name=study.name)
            
            
    def test_qsm_de_pipeline(self):
        study = self.create_study(
            T2StarStudy, 'qsm', input_datasets={
                'coils': Dataset('swi_coils_de', zip_format)})
        study.qsm_de_pipeline().run(work_dir=self.work_dir)
        for fname in ('qsm.nii.gz', 'tissue_phase.nii.gz',
                      'tissue_mask.nii.gz', 'qsm_mask.nii.gz'):
            self.assertDatasetCreated(dataset_name=fname,
>>>>>>> QSM DE
                                      study_name=study.name)        
