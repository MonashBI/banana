#!/usr/bin/env python
import tempfile
from nipype import config
config.enable_debug_mode()
import os.path  # @IgnorePep8
from nianalysis.dataset import Dataset  # @IgnorePep8
from nianalysis.data_formats import nifti_gz_format  # @IgnorePep8
from nianalysis.study.mri.base import MRStudy  # @IgnorePep8
from nianalysis.archive.local import LocalArchive  # @IgnorePep8
if __name__ == '__main__':
    from nianalysis.testing import DummyTestCase as TestCase  # @IgnorePep8 @UnusedImport
else:
    from nianalysis.testing import BaseImageTestCase as TestCase  # @IgnorePep8 @Reimport


class TestMR(TestCase):

    STUDY_NAME = 'mr'
    PROJECT_NAME = 'MR'

    def setUp(self):
        self.work_dir = tempfile.mkdtemp()

    def test_brain_mask(self):
        self._remove_generated_files(self.PROJECT_NAME)
        study = MRStudy(
            name=self.STUDY_NAME,
            project_id=self.EXAMPLE_INPUT_PROJECT,
            archive=LocalArchive(self.ARCHIVE_PATH),
            input_datasets={
                'primary': Dataset('mri_scan', nifti_gz_format)})
        study.brain_mask_pipeline().run(work_dir=self.work_dir)
        print self._session_dir(self.EXAMPLE_INPUT_PROJECT)
        self.assert_(
            os.path.exists(os.path.join(
                self._session_dir(self.EXAMPLE_INPUT_PROJECT),
                '{}_brain_mask.nii.gz'.format(self.STUDY_NAME))))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tester', default='diffusion', type=str,
                        help="Which tester to run the test from")
    parser.add_argument('--test', default='brain_mask', type=str,
                        help="Which test to run")
    args = parser.parse_args()
    tester = TestMR()
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
