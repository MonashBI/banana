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
    from nianalysis.testing import BaseImageTestCase as TestCase  # @IgnorePep8 @Reimport

from nianalysis.data_formats import coils_zip_format  # @IgnorePep8
from nianalysis.study.mri.structural.qsm import QSMStudy  # @IgnorePep8

logger = logging.getLogger('NiAnalysis')


class TestQSM(TestCase):

    STUDY_NAME = 'qsm'
    PROJECT_NAME = 'QSM'
    ARCHIVE_PATH = os.path.join(test_data_dir, 'study', 'mri', 'structural',
                                'qsm')
    WORK_DIR = os.path.join(test_data_dir, 'work', 'qsm')
    XNAT_CACHE_PATH = os.path.join(test_data_dir, 'cache', 'xnat')

    XNAT_URL = 'https://mbi-xnat.erc.monash.edu.au'
    XNAT_LOGIN = 'unittest'
    XNAT_PASSWORD = 'Test123!'
    PROJECT_ID = 'TEST001'
    SUBJECT_ID = 'TESTQSM'
    SESSION_ID = 'MR01'

    COILS_NAME = 'swi_coils'
    COILS_DF = 'COILS_ZIP'
    COILS_EXT = '.zip'

    def setUp(self):
        shutil.rmtree(self.WORK_DIR, ignore_errors=True)
        os.makedirs(self.WORK_DIR)
        shutil.rmtree(self.session_dir, ignore_errors=True)
        # Copy swi_coils.zip from XNAT to new archive
        os.makedirs(self.session_dir)
        cache_dir = os.path.join(self.XNAT_CACHE_PATH, self.xnat_sess_id)
        try:
            os.makedirs(cache_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        cache_path = os.path.join(cache_dir, self.coils_fname)
        if not os.path.exists(cache_path):
            with xnat.connect(self.XNAT_URL, self.XNAT_LOGIN,
                              self.XNAT_PASSWORD) as login:
                tmp_dir = cache_path + '.download'
                shutil.rmtree(tmp_dir, ignore_errors=True)
                login.experiments[self.xnat_sess_id].scans[
                    self.COILS_NAME].resources[self.COILS_DF].download_dir(
                        tmp_dir)
                shutil.move(os.path.join(tmp_dir, self.xnat_sess_id, 'scans',
                                         '{}-{}'.format(self.COILS_NAME,
                                                        self.COILS_NAME),
                                         'resources', self.COILS_DF, 'files',
                                         self.coils_fname),
                            cache_path)
                shutil.rmtree(tmp_dir, ignore_errors=True)
        shutil.copy(cache_path,
                    os.path.join(self.session_dir, self.coils_fname))
        print os.listdir(self.session_dir)

    @property
    def xnat_sess_id(self):
        return '{}_{}_{}'.format(self.PROJECT_ID, self.SUBJECT_ID,
                                 self.SESSION_ID)

    @property
    def session_dir(self):
        return os.path.join(self.ARCHIVE_PATH, self.PROJECT_NAME,
                            'SUBJECT1', 'SESSION1')

    @property
    def coils_fname(self):
        return self.COILS_NAME + self.COILS_EXT

    def test_qsm_pipeline(self):
        self._remove_generated_files(self.PROJECT_NAME)
        study = QSMStudy(
            name=self.DATASET_NAME,
            project_id=self.PROJECT_NAME,
            archive=LocalArchive(self.ARCHIVE_PATH),
            input_datasets={
                'swi_coils': Dataset('swi_coils', coils_zip_format)})
        study.qsm_pipeline().run(work_dir=self.WORK_DIR)
        output_path = os.path.join(self.session_dir, 'qsm.nii.gz')
        self.assert_(os.path.exists(output_path),
                     "Output path '{}' was not created".format(output_path))
