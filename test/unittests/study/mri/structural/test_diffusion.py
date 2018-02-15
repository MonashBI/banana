#!/usr/bin/env python
from nipype import config
config.enable_debug_mode()
from nianalysis.dataset import Dataset  # @IgnorePep8
from mbianalysis.study.mri.structural.diffusion import (  # @IgnorePep8
    DiffusionStudy, NODDIStudy)
from nianalysis.data_formats import (  # @IgnorePep8
    mrtrix_format, nifti_gz_format, fsl_bvals_format, fsl_bvecs_format,
    text_format)
from nianalysis.testing import BaseTestCase, BaseMultiSubjectTestCase  # @IgnorePep8 @Reimport


class TestDiffusion(BaseTestCase):

    def test_preprocess(self):
        study = self.create_study(
            DiffusionStudy, 'preprocess', {
                'dwi_scan': Dataset('r_l_dwi_b700_30',
                                    mrtrix_format),
                'reverse_pe': Dataset('l_r_dwi_b0_6', mrtrix_format)})
        study.preprocess_pipeline(preproc_pe_dir='RL',
                                  preproc_denoise=True).run(
            work_dir=self.work_dir)
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
        study.bias_correct_pipeline(mask_tool='mrtrix').run(
            work_dir=self.work_dir)
        self.assertDatasetCreated('bias_correct.nii.gz', study.name)

    def test_tensor(self):
        study = self.create_study(
            DiffusionStudy, 'tensor', {
                'bias_correct': Dataset('bias_correct', nifti_gz_format),
                'brain_mask': Dataset('brain_mask', nifti_gz_format),
                'grad_dirs': Dataset('gradient_dirs', fsl_bvecs_format),
                'bvalues': Dataset('bvalues', fsl_bvals_format)})
        study.tensor_pipeline().run(
            work_dir=self.work_dir)
        self.assertDatasetCreated('tensor.nii.gz', study.name)

    def test_response(self):
        study = self.create_study(
            DiffusionStudy, 'response', {
                'bias_correct': Dataset('bias_correct', nifti_gz_format),
                'brain_mask': Dataset('brain_mask', nifti_gz_format),
                'grad_dirs': Dataset('gradient_dirs', fsl_bvecs_format),
                'bvalues': Dataset('bvalues', fsl_bvals_format)})
        study.response_pipeline().run(
            work_dir=self.work_dir)
        self.assertDatasetCreated('response.txt', study.name)

    def test_fod(self):
        study = self.create_study(
            DiffusionStudy, 'fod', {
                'bias_correct': Dataset('bias_correct', nifti_gz_format),
                'brain_mask': Dataset('brain_mask', nifti_gz_format),
                'grad_dirs': Dataset('gradient_dirs', fsl_bvecs_format),
                'response': Dataset('response', text_format),
                'bvalues': Dataset('bvalues', fsl_bvals_format)})
        study.fod_pipeline().run(
            work_dir=self.work_dir)
        self.assertDatasetCreated('fod.mif', study.name)


class TestMultiSubjectDiffusion(BaseMultiSubjectTestCase):

    def test_intensity_normalization(self):
        study = self.create_study(
            DiffusionStudy, 'intens_norm', {
                'bias_correct': Dataset('biascorrect', nifti_gz_format),
                'brain_mask': Dataset('brainmask', nifti_gz_format),
                'grad_dirs': Dataset('gradientdirs', fsl_bvecs_format),
                'bvalues': Dataset('bvalues', fsl_bvals_format)})
        study.intensity_normalisation_pipeline().run(
            work_dir=self.work_dir)
        for subject_id in self.subject_ids:
            for visit_id in self.visit_ids(subject_id):
                self.assertDatasetCreated('norm_intensity.mif', study.name,
                                          subject=subject_id, visit=visit_id)
        self.assertDatasetCreated(
            'norm_intens_fa_template.mif', study.name,
            multiplicity='per_project')
        self.assertDatasetCreated(
            'norm_intens_wm_mask.mif', study.name,
            multiplicity='per_project')

    def test_average_response(self):
        study = self.create_study(
            DiffusionStudy, 'response', {
                'response': Dataset('response', text_format)})
        study.average_response_pipeline().run(work_dir=self.work_dir)
        for subject_id in self.subject_ids:
            for visit_id in self.visit_ids(subject_id):
                self.assertDatasetCreated('avg_response.txt', study.name,
                                          subject=subject_id, visit=visit_id)


class TestNODDI(BaseTestCase):

    def test_concatenate(self):
        study = self.create_study(
            NODDIStudy, 'concatenate', inputs={
                'low_b_dw_scan': Dataset('r_l_dwi_b700_30', mrtrix_format),
                'high_b_dw_scan': Dataset('r_l_dwi_b2000_60', mrtrix_format)})
        study.concatenate_pipeline().run(work_dir=self.work_dir)
        self.assertDatasetCreated('dwi_scan.mif', study.name)
        # TODO: More thorough testing required

#     def test_noddi_fitting(self, nthreads=6):
#         study = self.create_study(
#             NODDIStudy, 'noddi', inputs={
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
