#!/usr/bin/env python
from nipype import config
config.enable_debug_mode()
from nianalysis.dataset import Dataset  # @IgnorePep8
from nianalysis.study.mri.structural.diffusion import (  # @IgnorePep8
    DiffusionStudy, NODDIStudy)
from nianalysis.data_formats import (  # @IgnorePep8
    mrtrix_format, nifti_gz_format, fsl_bvals_format, fsl_bvecs_format)
from nianalysis.testing import PipelineTestCase as TestCase  # @IgnorePep8 @Reimport


class TestDiffusion(TestCase):

    def test_preprocess(self):
        study = self.create_study(
            DiffusionStudy, 'preprocess', {
                'dwi_scan': Dataset('r_l_dwi_b700_30',
                                    mrtrix_format),
                'forward_rpe': Dataset('r_l_dwi_b0_6', mrtrix_format),
                'reverse_rpe': Dataset('l_r_dwi_b0_6', mrtrix_format)})
        study.preprocess_pipeline().run(work_dir=self.work_dir)
        self.assertDatasetCreated('dwi_preproc.nii.gz', study.name)

    def test_extract_b0(self):
        study = self.create_study(
            DiffusionStudy, 'extract_b0', {
                'dwi_preproc': Dataset('dwi_preproc', nifti_gz_format),
                'grad_dirs': Dataset('gradient_dirs',
                                     fsl_bvecs_format),
                'bvalues': Dataset('bvalues', fsl_bvals_format)})
        study.extract_b0_pipeline().run(work_dir=self.work_dir)
        self.assertDatasetCreated('primary.nii.gz', study.name)

    def test_bias_correct(self):
        study = self.create_study(
            DiffusionStudy, 'bias_correct', {
                'dwi_preproc': Dataset('dwi_preproc', nifti_gz_format),
                'grad_dirs': Dataset('gradient_dirs', fsl_bvecs_format),
                'bvalues': Dataset('bvalues', fsl_bvals_format)})
        study.bias_correct_pipeline(mask_tool='dwi2mask').run(
            work_dir=self.work_dir)
        self.assertDatasetCreated('bias_correct.nii.gz', study.name)


class TestNODDI(TestCase):

    def test_concatenate(self):
        study = self.create_study(
            NODDIStudy, 'concatenate', input_datasets={
                'low_b_dw_scan': Dataset('r_l_noddi_b700_30_directions',
                                         mrtrix_format),
                'high_b_dw_scan': Dataset('r_l_noddi_b2000_60_directions',
                                          mrtrix_format)})
        study.concatenate_pipeline().run(work_dir=self.work_dir)
        self.assertDatasetCreated('dwi.mif')
        # TODO: More thorough testing required

#     def test_noddi_fitting(self, nthreads=6):
#         study = self.create_study(
#             NODDIStudy, 'noddi', input_datasets={
#                 'dwi_preproc': Dataset('noddi_dwi', mrtrix_format),
#                 'brain_mask': Dataset('roi_mask', analyze_format),
#                 'grad_dirs': Dataset('noddi_gradient_directions',
#                                      fsl_bvecs_format),
#                 'bvalues': Dataset('noddi_bvalues', fsl_bvals_format)})
#         study.noddi_fitting_pipeline(nthreads=nthreads).run(
#             work_dir=self.work_dir)
#         for out_name, mean, stdev in [('ficvf', 1e-5, 1e-2),
#                                       ('odi', 1e-4, 1e-2),
#                                       ('fiso', 1e-4, 1e-2),
#                                       ('fibredirs_xvec', 1e-3, 1e-1),
#                                       ('fibredirs_yvec', 1e-3, 1e-1),
#                                       ('fibredirs_zvec', 1e-3, 1e-1),
#                                       ('kappa', 1e-4, 1e-1)]:
#             self.assertImagesAlmostMatch(
#                 'example_{}.nii'.format(out_name), (out_name + '.nii'),
#                 mean_threshold=mean, stdev_threshold=stdev)
