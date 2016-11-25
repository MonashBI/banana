#!/usr/bin/env python
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


class TestCoregisteredStudy(TestCase):

    DATASET_NAME = 'coreg'
    PROJECT_NAME = 't1t2'

    def test_registration(self):
        self._remove_generated_files(self.PROJECT_NAME)
        study = CoregisteredStudy(
            name=self.DATASET_NAME,
            project_id=self.PROJECT_NAME,
            archive=LocalArchive(self.ARCHIVE_PATH),
            input_datasets={
                'to_register': Dataset('t2', nifti_gz_format),
                'reference': Dataset('t1', nifti_gz_format)})
        pipeline = study.registration_pipeline()
        pipeline.run()
        self.assertTrue(
            os.path.exists(os.path.join(
                self._session_dir(self.PROJECT_NAME),
                'registered{}'.format(nifti_gz_format.extension))),
            "'registered{}' was not created by registration pipeline"
            .format(nifti_gz_format.extension))
        self.assertTrue(
            os.path.exists(os.path.join(
                self._session_dir(self.PROJECT_NAME),
                'matrix{}'.format(text_matrix_format.extension))),
            "'matrix{}' was not created by registration pipeline"
            .format(text_matrix_format.extension))
        self._remove_generated_files(self.PROJECT_NAME)
        # Temporary
        os.remove(os.path.join(
            self._session_dir(self.PROJECT_NAME), 'registered.nii.gz'))
        to_matrix_study = CoregisteredToMatrixStudy(
            name=self.DATASET_NAME + '_to_matrix',
            project_id=self.PROJECT_NAME,
            archive=LocalArchive(self.ARCHIVE_PATH),
            input_datasets={
                'to_register': Dataset('t2', nifti_gz_format),
                'reference': Dataset('t1', nifti_gz_format),
                'matrix': Dataset('matrix', text_matrix_format)})
        to_matrix_study.registration_pipeline().run()
        self.assertTrue(
            os.path.exists(os.path.join(
                self._session_dir(self.PROJECT_NAME),
                'registered{}'.format(nifti_gz_format.extension))),
            "'registered{}' was not created by registration pipeline"
            .format(nifti_gz_format.extension))


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
