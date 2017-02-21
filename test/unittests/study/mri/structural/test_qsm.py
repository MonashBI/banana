import shutil
import errno
import logging  # @IgnorePep8
from nipype import config
config.enable_debug_mode()
import xnat  # @IgnorePep8
import os.path  # @IgnorePep8
from nianalysis.dataset import Dataset  # @IgnorePep8
from nianalysis.archive.local import LocalArchive  # @IgnorePep8
from nianalysis.testing import test_data_dir  # @IgnorePep8
if __name__ == '__main__':
    from nianalysis.testing import DummyTestCase as TestCase  # @IgnorePep8 @UnusedImport
else:
    from nianalysis.testing import PipelineTeseCase as TestCase  # @IgnorePep8 @Reimport

from nianalysis.data_formats import coils_zip_format  # @IgnorePep8
from nianalysis.study.mri.structural.qsm import QSMStudy  # @IgnorePep8

logger = logging.getLogger('NiAnalysis')


class TestQSM(TestCase):

    TEST_MODULE = 'STUDYMRISTRUCTURALQSM'
    TEST_NAME = 'QSM'
    REQUIRED_DATASETS = ['swi_coils.zip']

    def test_qsm_pipeline(self):
        study = self.create_study(QSMStudy, 'qsm', {
            'swi_coils': Dataset('swi_coils', coils_zip_format)})
        study.qsm_pipeline().run(work_dir=self.work_dir)
        self.assertDatasetCreated('qsm.nii.gz')
