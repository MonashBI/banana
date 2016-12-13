#!/usr/bin/env python
import tempfile
import shutil
from nipype import config
config.enable_debug_mode()
import os.path  # @IgnorePep8
from nianalysis.dataset import Dataset  # @IgnorePep8
from nianalysis.data_formats import nifti_gz_format, text_matrix_format  # @IgnorePep8
from nianalysis.study.mri.coregistered import (  # @IgnorePep8
    CoregisteredStudy, CoregisteredToMatrixStudy)
from nianalysis.archive.local import LocalArchive  # @IgnorePep8
if __name__ == '__main__':
    from nianalysis.testing import DummyTestCase as TestCase  # @IgnorePep8 @UnusedImport
else:
    from nianalysis.testing import BaseImageTestCase as TestCase  # @IgnorePep8 @Reimport
from nianalysis.testing import test_data_dir  # @IgnorePep8


class TestCoregisteredStudy(TestCase):

    STUDY_NAME = 'coreg'
    PROJECT_NAME = 't1t2'
    WORK_DIR = os.path.join(test_data_dir, 'work', 'test_coreg')

    def setUp(self):
        shutil.rmtree(self.WORK_DIR, ignore_errors=True)
        os.mkdir(self.WORK_DIR)
        shutil.rmtree(self.ARCHIVE_PATH, ignore_errors=True)
        if not os.path.exists(self._session_dir(self.PROJECT_NAME)):
            os.makedirs(self._session_dir(self.PROJECT_NAME))
            for fname in ('mprage.nii.gz', 'flair.nii.gz'):
                shutil.copy(os.path.join(test_data_dir, fname),
                            os.path.join(self._session_dir(self.PROJECT_NAME),
                                         fname))

    def test_registration(self):
        self._remove_generated_files(self.PROJECT_NAME)
        study = CoregisteredStudy(
            name=self.STUDY_NAME,
            project_id=self.PROJECT_NAME,
            archive=LocalArchive(self.ARCHIVE_PATH),
            input_datasets={
                'to_register': Dataset('flair', nifti_gz_format),
                'reference': Dataset('mprage', nifti_gz_format)})
        pipeline = study.registration_pipeline()
        pipeline.run(work_dir=self.WORK_DIR)
        reg_path = os.path.join(
            self._session_dir(self.PROJECT_NAME),
            '{}_registered{}'.format(self.STUDY_NAME,
                                     nifti_gz_format.extension))
        self.assertTrue(
            os.path.exists(reg_path),
            "'{}' was not created by registration pipeline".format(reg_path))
        mat_path = os.path.join(
            self._session_dir(self.PROJECT_NAME),
            '{}_matrix{}'.format(self.STUDY_NAME,
                                 text_matrix_format.extension))
        self.assertTrue(os.path.exists(mat_path),
                        "'{}' was not created by registration pipeline".format(
                            mat_path))
        # Move the generated matrix file to a location that won't be cleaned
        shutil.move(mat_path,
                    os.path.join(self._session_dir(self.PROJECT_NAME),
                                 'matrix.mat'))
        self._remove_generated_files(self.PROJECT_NAME)
        to_matrix_study = CoregisteredToMatrixStudy(
            name=self.STUDY_NAME + 'wmat',
            project_id=self.PROJECT_NAME,
            archive=LocalArchive(self.ARCHIVE_PATH),
            input_datasets={
                'to_register': Dataset('flair', nifti_gz_format),
                'reference': Dataset('mprage', nifti_gz_format),
                'matrix': Dataset('matrix', text_matrix_format)})
        to_matrix_study.registration_pipeline().run(work_dir=self.WORK_DIR)
        matreg_path = os.path.join(
            self._session_dir(self.PROJECT_NAME),
            '{}_registered{}'.format(self.STUDY_NAME + 'wmat',
                                     nifti_gz_format.extension))
        self.assertTrue(
            os.path.exists(matreg_path),
            "'{}' was not created by registration pipeline"
            .format(matreg_path))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tester', default='diffusion', type=str,
                        help="Which tester to run the test from")
    parser.add_argument('--test', default='registration', type=str,
                        help="Which test to run")
    args = parser.parse_args()
    tester = TestCoregisteredStudy()
    tester.setUp()
    try:
        getattr(tester, 'test_' + args.test)()
    except AttributeError as e:
        if str(e) == 'test_' + args.test:
            raise Exception("Unrecognised test '{}' for '{}' tester"
                            .format(args.test, args.tester))
        else:
            raise
    finally:
        tester.tearDown()
