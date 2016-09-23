#!/usr/bin/env python

import os.path  # @IgnorePep8
import shutil
from nipype import config
config.enable_debug_mode()
from nianalysis.base import Scan  # @IgnorePep8
from nianalysis.dataset.mri import T1AndT2Dataset # @IgnorePep8
from nianalysis.archive import LocalArchive  # @IgnorePep8
from nianalysis.formats import (  # @IgnorePep8
    nifti_format)
if __name__ == '__main__':
    from nianalysis.testing import DummyTestCase as TestCase  # @IgnorePep8 @UnusedImport
else:
    from nianalysis.testing import BaseImageTestCase as TestCase  # @IgnorePep8 @Reimport
from nianalysis.testing import test_data_dir  # @IgnorePep8


class TestT1AndT2(TestCase):

    DATASET_NAME = 'T1AndT2'
    PROJECT_NAME = 'T1AndT2'
    ARCHIVE_PATH = os.path.join(test_data_dir, 'archives', 't1_and_t2')

    def setUp(self):
        shutil.rmtree(self.ARCHIVE_PATH, ignore_errors=True)
        os.makedirs(self._session_dir(self.PROJECT_NAME))
        for fname in ('t1.nii', 't2.nii'):
            shutil.copy(os.path.join(test_data_dir, fname),
                        os.path.join(self._session_dir(self.PROJECT_NAME),
                                     fname))

    def tearDown(self):
#         shutil.rmtree(self.ARCHIVE_PATH, ignore_errors=True)
        pass

    def test_coregistration_pipeline(self):
        self._remove_generated_files(self.PROJECT_NAME)
        dataset = T1AndT2Dataset(
            name=self.DATASET_NAME,
            project_id=self.PROJECT_NAME,
            archive=LocalArchive(self.ARCHIVE_PATH),
            input_scans={
                't1': Scan('t1', nifti_format),
                't2': Scan('t2', nifti_format)})
        dataset.coregistration_pipeline().run()
        self.assert_(
            os.path.exists(os.path.join(
                self._session_dir(self.PROJECT_NAME),
                't2_coreg.nii'.format(self.DATASET_NAME))))

if __name__ == '__main__':
    tester = TestT1AndT2()
    tester.test_coregistration_pipeline()