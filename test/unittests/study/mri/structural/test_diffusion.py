#!/usr/bin/env python
import shutil
from nipype import config
config.enable_debug_mode()
import os.path  # @IgnorePep8
from nianalysis.dataset import Dataset  # @IgnorePep8
from nianalysis.study.mri.structural.diffusion import (  # @IgnorePep8
    DiffusionStudy, NODDIStudy)
from nianalysis.archive.local import LocalArchive  # @IgnorePep8
from nianalysis.data_formats import (  # @IgnorePep8
    mrtrix_format, nifti_gz_format, fsl_bvals_format, fsl_bvecs_format)
if __name__ == '__main__':
    from nianalysis.testing import DummyTestCase as TestCase  # @IgnorePep8 @UnusedImport
else:
    from nianalysis.testing import BaseImageTestCase as TestCase  # @IgnorePep8 @Reimport
from nianalysis.testing import test_data_dir  # @IgnorePep8


class TestDiffusion(TestCase):

    STUDY_NAME = 'diff'
    WORK_DIR = os.path.join(test_data_dir, 'work', 'diffusion')

    def setUp(self):
        shutil.rmtree(self.WORK_DIR, ignore_errors=True)
        os.mkdir(self.WORK_DIR)

    def test_preprocess(self):
        self._remove_generated_files(self.PILOT_PROJECT)
        study = DiffusionStudy(
            name=self.STUDY_NAME,
            project_id=self.PILOT_PROJECT,
            archive=LocalArchive(self.ARCHIVE_PATH),
            input_datasets={
                'dwi_scan': Dataset('r_l_noddi_b700_30_directions',
                                    mrtrix_format),
                'forward_rpe': Dataset('r_l_noddi_b0_6', mrtrix_format),
                'reverse_rpe': Dataset('l_r_noddi_b0_6', mrtrix_format)})
        study.preprocess_pipeline().run(work_dir=self.WORK_DIR)
        self.assert_(
            os.path.exists(os.path.join(
                self._session_dir(self.PILOT_PROJECT),
                '{}_dwi_preproc.mif'.format(self.STUDY_NAME))))

    def test_extract_b0(self):
        self._remove_generated_files(self.PILOT_PROJECT)
        study = DiffusionStudy(
            name=self.STUDY_NAME,
            project_id=self.PILOT_PROJECT,
            archive=LocalArchive(self.ARCHIVE_PATH),
            input_datasets={
                'dwi_preproc': Dataset('noddi_dwi', mrtrix_format),
                'grad_dirs': Dataset('noddi_gradient_directions',
                                     fsl_bvecs_format),
                'bvalues': Dataset('noddi_bvalues', fsl_bvals_format)})
        study.extract_b0_pipeline().run(work_dir=self.WORK_DIR)
        self.assert_(
            os.path.exists(os.path.join(
                self._session_dir(self.PILOT_PROJECT),
                '{}_primary.nii.gz'.format(self.STUDY_NAME))))

    def test_bias_correct(self):
        self._remove_generated_files(self.PILOT_PROJECT)
        study = DiffusionStudy(
            name=self.STUDY_NAME,
            project_id=self.PILOT_PROJECT,
            archive=LocalArchive(self.ARCHIVE_PATH),
            input_datasets={
                'dwi_preproc': Dataset('dwi_preproc', nifti_gz_format),
                'grad_dirs': Dataset('noddi_gradient_directions',
                                     fsl_bvecs_format),
                'bvalues': Dataset('noddi_bvalues', fsl_bvals_format)})
        study.bias_correct_pipeline(mask_tool='dwi2mask').run(
            work_dir=self.WORK_DIR)
        self.assert_(
            os.path.exists(os.path.join(
                self._session_dir(self.PILOT_PROJECT),
                '{}_bias_correct.nii.gz'.format(self.STUDY_NAME))))


class TestNODDI(TestCase):

    STUDY_NAME = 'NODDI'
    WORK_DIR = os.path.join(test_data_dir, 'work', 'noddi')

    def test_concatenate(self):
        self._remove_generated_files(self.PILOT_PROJECT)
        study = NODDIStudy(
            name=self.STUDY_NAME,
            project_id=self.PILOT_PROJECT,
            archive=LocalArchive(self.ARCHIVE_PATH),
            input_datasets={
                'low_b_dw_scan': Dataset('r_l_noddi_b700_30_directions',
                                         mrtrix_format),
                'high_b_dw_scan': Dataset('r_l_noddi_b2000_60_directions',
                                          mrtrix_format)})
        study.concatenate_pipeline().run(work_dir=self.WORK_DIR)
        self.assert_(
            os.path.exists(os.path.join(
                self._session_dir(self.PILOT_PROJECT),
                '{}_dwi.mif'.format(self.STUDY_NAME))),
            "Concatenated file was not created")
        # TODO: More thorough testing required

#     def test_noddi_fitting(self, nthreads=6):
#         self._remove_generated_files(self.PILOT_PROJECT)
#         study = NODDIStudy(
#             name=self.STUDY_NAME,
#             project_id=self.PILOT_PROJECT,
#             archive=LocalArchive(self.ARCHIVE_PATH),
#             input_datasets={
#                 'dwi_preproc': Dataset('noddi_dwi', mrtrix_format),
#                 'brain_mask': Dataset('roi_mask', analyze_format),
#                 'grad_dirs': Dataset('noddi_gradient_directions',
#                                      fsl_bvecs_format),
#                 'bvalues': Dataset('noddi_bvalues', fsl_bvals_format)})
#         study.noddi_fitting_pipeline(nthreads=nthreads).run(
#             work_dir=self.WORK_DIR)
#         ref_out_path = os.path.join(
#             self.ARCHIVE_PATH, self.PILOT_PROJECT, self.SUBJECT,
#             self.SESSION)
#         gen_out_path = os.path.join(
#             self.ARCHIVE_PATH, self.PILOT_PROJECT, self.SUBJECT,
#             self.SESSION)
#         for out_name, mean, stdev in [('ficvf', 1e-5, 1e-2),
#                                       ('odi', 1e-4, 1e-2),
#                                       ('fiso', 1e-4, 1e-2),
#                                       ('fibredirs_xvec', 1e-3, 1e-1),
#                                       ('fibredirs_yvec', 1e-3, 1e-1),
#                                       ('fibredirs_zvec', 1e-3, 1e-1),
#                                       ('kappa', 1e-4, 1e-1)]:
#             self.assertImagesAlmostMatch(
#                 os.path.join(ref_out_path, 'example_{}.nii'.format(out_name)),
#                 os.path.join(gen_out_path,
#                              '{}_{}.nii'.format(self.STUDY_NAME, out_name)),
#                 mean_threshold=mean, stdev_threshold=stdev)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tester', default='diffusion', type=str,
                        help="Which tester to run the test from")
    parser.add_argument('--test', default='preprocess', type=str,
                        help="Which test to run")
    args = parser.parse_args()
    if args.tester == 'diffusion':
        tester = TestDiffusion()
    elif args.tester == 'noddi':
        tester = TestNODDI()
    else:
        raise Exception("Unrecognised tester '{}'")
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
