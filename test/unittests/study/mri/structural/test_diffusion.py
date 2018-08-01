#!/usr/bin/env python
import os.path
from nipype import config
config.enable_debug_mode()
from arcana.data import FilesetMatch  # @IgnorePep8
from nianalysis.study.mri.structural.diffusion import (  # @IgnorePep8
    DiffusionStudy, NODDIStudy)
from nianalysis.file_format import (  # @IgnorePep8
    mrtrix_format, nifti_gz_format, fsl_bvals_format, fsl_bvecs_format,
    text_format)
from nianalysis.testing import BaseTestCase, BaseMultiSubjectTestCase  # @IgnorePep8 @Reimport


class TestDiffusion(BaseTestCase):

    def test_preprocess(self):
        study = self.create_study(
            DiffusionStudy, 'preprocess', [
                FilesetMatch('primary', mrtrix_format, 'r_l_dwi_b700_30'),
                FilesetMatch('dwi_reference', mrtrix_format, 'l_r_dwi_b0_6')])
        preproc = list(study.data('preproc'))[0]
        self.assertTrue(os.path.exists(preproc.path))

    def test_extract_b0(self):
        study = self.create_study(
            DiffusionStudy, 'extract_b0', [
                FilesetMatch('preproc', nifti_gz_format, 'preproc'),
                FilesetMatch('grad_dirs', fsl_bvecs_format, 'gradient_dirs'),
                FilesetMatch('bvalues', fsl_bvals_format, 'bvalues')])
        study.extract_b0_pipeline().run(work_dir=self.work_dir)
        self.assertFilesetCreated('primary.nii.gz', study.name)

    def test_bias_correct(self):
        study = self.create_study(
            DiffusionStudy, 'bias_correct', [
                FilesetMatch('preproc', nifti_gz_format, 'preproc'),
                FilesetMatch('grad_dirs', fsl_bvecs_format, 'gradient_dirs'),
                FilesetMatch('bvalues', fsl_bvals_format, 'bvalues')])
        study.bias_correct_pipeline(mask_tool='mrtrix').run(
            work_dir=self.work_dir)
        self.assertFilesetCreated('bias_correct.nii.gz', study.name)

    def test_tensor(self):
        study = self.create_study(
            DiffusionStudy, 'tensor', [
                FilesetMatch('bias_correct', nifti_gz_format, 'bias_correct'),
                FilesetMatch('brain_mask', nifti_gz_format, 'brain_mask'),
                FilesetMatch('grad_dirs', fsl_bvecs_format, 'gradient_dirs'),
                FilesetMatch('bvalues', fsl_bvals_format, 'bvalues')])
        study.tensor_pipeline().run(
            work_dir=self.work_dir)
        self.assertFilesetCreated('tensor.nii.gz', study.name)

    def test_response(self):
        study = self.create_study(
            DiffusionStudy, 'response', [
                FilesetMatch('bias_correct', nifti_gz_format, 'bias_correct'),
                FilesetMatch('brain_mask', nifti_gz_format, 'brain_mask'),
                FilesetMatch('grad_dirs', fsl_bvecs_format, 'gradient_dirs'),
                FilesetMatch('bvalues', fsl_bvals_format, 'bvalues')])
        study.response_pipeline().run(
            work_dir=self.work_dir)
        self.assertFilesetCreated('response.txt', study.name)

    def test_fod(self):
        study = self.create_study(
            DiffusionStudy, 'fod', [
                FilesetMatch('bias_correct', nifti_gz_format, 'bias_correct'),
                FilesetMatch('brain_mask', nifti_gz_format, 'brain_mask'),
                FilesetMatch('grad_dirs', fsl_bvecs_format, 'gradient_dirs'),
                FilesetMatch('response', text_format, 'response'),
                FilesetMatch('bvalues', fsl_bvals_format, 'bvalues')])
        study.fod_pipeline().run(
            work_dir=self.work_dir)
        self.assertFilesetCreated('fod.mif', study.name)


class TestMultiSubjectDiffusion(BaseMultiSubjectTestCase):

    def test_intensity_normalization(self):
        study = self.create_study(
            DiffusionStudy, 'intens_norm', [
                FilesetMatch('bias_correct', nifti_gz_format, 'biascorrect'),
                FilesetMatch('brain_mask', nifti_gz_format, 'brainmask'),
                FilesetMatch('grad_dirs', fsl_bvecs_format, 'gradientdirs'),
                FilesetMatch('bvalues', fsl_bvals_format, 'bvalues')])
        study.intensity_normalisation_pipeline().run(
            work_dir=self.work_dir)
        for subject_id in self.subject_ids:
            for visit_id in self.visit_ids(subject_id):
                self.assertFilesetCreated('norm_intensity.mif', study.name,
                                          subject=subject_id, visit=visit_id)
        self.assertFilesetCreated(
            'norm_intens_fa_template.mif', study.name,
            frequency='per_study')
        self.assertFilesetCreated(
            'norm_intens_wm_mask.mif', study.name,
            frequency='per_study')

    def test_average_response(self):
        study = self.create_study(
            DiffusionStudy, 'response', {
                FilesetMatch('response', text_format, 'response')})
        study.average_response_pipeline().run(work_dir=self.work_dir)
        for subject_id in self.subject_ids:
            for visit_id in self.visit_ids(subject_id):
                self.assertFilesetCreated('avg_response.txt', study.name,
                                          subject=subject_id, visit=visit_id)


class TestNODDI(BaseTestCase):

    def test_concatenate(self):
        study = self.create_study(
            NODDIStudy, 'concatenate', inputs=[
                FilesetMatch('low_b_dw_scan', mrtrix_format, 'r_l_dwi_b700_30'),
                FilesetMatch('high_b_dw_scan', mrtrix_format, 'r_l_dwi_b2000_60')])
        study.concatenate_pipeline().run(work_dir=self.work_dir)
        self.assertFilesetCreated('dwi_scan.mif', study.name)
        # TODO: More thorough testing required

#     def test_noddi_fitting(self, nthreads=6):
#         study = self.create_study(
#             NODDIStudy, 'noddi', inputs=[
#                 FilesetMatch('preproc', mrtrix_format, 'noddi_dwi'),
#                 FilesetMatch('brain_mask', analyze_format, 'roi_mask'),
#                 'grad_dirs': Fileset('noddi_gradient_directions',
#                                      fsl_bvecs_format),
#                 FilesetMatch('bvalues', fsl_bvals_format, 'noddi_bvalues')})
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
