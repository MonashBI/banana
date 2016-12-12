from nipype.pipeline import engine as pe
from nipype.interfaces.fsl import ApplyMask
from nianalysis.data_formats import (
    nifti_gz_format, freesurfer_recon_all_format, text_matrix_format)
from nianalysis.study.base import set_dataset_specs
from nianalysis.dataset import DatasetSpec
from ...combined import CombinedStudy
from ..coregistered import CoregisteredStudy, CoregisteredToMatrixStudy
from .t1 import T1Study
from .t2 import T2Study
from nianalysis.requirements import fsl5_req
from nianalysis.citations import fsl_cite


class T1T2Study(CombinedStudy):
    """
    T1 and T2 weighted MR dataset, with the T2-weighted coregistered to the T1.
    """

    sub_study_specs = {
        't1_study': (T1Study, {
            't1': 'acquired',
            'fs_recon_all': 'fs_recon_all'}),
        't2_study': (T2Study, {
            't2_coreg': 'acquired',
            'manual_wmh_mask_coreg': 'manual_wmh_mask',
            't2_masked': 'masked',
            'brain_mask': 'brain_mask'}),
        'coreg_t2_to_t1_study': (CoregisteredStudy, {
            't1': 'reference',
            't2': 'to_register',
            't2_coreg': 'registered',
            't2_coreg_matrix': 'matrix'}),
        'coreg_manual_wmh_mask_study': (CoregisteredToMatrixStudy, {
            't1': 'reference',
            'manual_wmh_mask': 'to_register',
            't2_coreg_matrix': 'matrix',
            'manual_wmh_mask_coreg': 'registered'})}

    def freesurfer_pipeline(self, **kwargs):
        pipeline = self.TranslatedPipeline(
            'freesurfer', self.t1_study.freesurfer_pipeline(**kwargs), self,
            add_inputs=['t2_coreg'])
        recon_all = pipeline.node('recon_all')
        recon_all.inputs.use_T2 = True
        # Connect T2-weighted input
        pipeline.connect_input('t2_coreg', recon_all, 'T2_file')
        pipeline.assert_connected()
        return pipeline

    t2_registration_pipeline = CombinedStudy.translate(
        'coreg_t2_to_t1_study', CoregisteredStudy.registration_pipeline)

    manual_wmh_mask_registration_pipeline = CombinedStudy.translate(
        'coreg_t2_to_t1_study',
        CoregisteredToMatrixStudy.registration_pipeline)

    t2_brain_mask_pipeline = CombinedStudy.translate(
        't2_study', T2Study.brain_mask_pipeline)

    def t1_brain_mask_pipeline(self):
        """
        Masks the T1 image using the coregistered T2 brain mask as the brain
        mask from T2 is usually more reliable (using BET in any case)
        """
        pipeline = self._create_pipeline(
            name='t1_brain_mask_pipeline',
            inputs=['t1', 'brain_mask'],
            outputs=['t1_masked'],
            options={},
            description="Mask T1 with T2 brain mask",
            requirements=[fsl5_req],
            citations=[fsl_cite],
            approx_runtime=1)
        # Create apply mask node
        apply_mask = pe.Node(ApplyMask(), 'appy_mask')
        # Connect inputs
        pipeline.connect_input('t1', apply_mask, 'in_file')
        pipeline.connect_input('brain_mask', apply_mask, 'mask_file')
        # Connect outputs
        pipeline.connect_output('t1_masked', apply_mask, 'out_file')
        # Check and return
        pipeline.assert_connected()
        return pipeline

    _dataset_specs = set_dataset_specs(
        DatasetSpec('t1', nifti_gz_format,
                    description="Raw T1-weighted image (e.g. MPRAGE)"),
        DatasetSpec('t2', nifti_gz_format,
                    description="Raw T2-weighted image (e.g. FLAIR)"),
        DatasetSpec('manual_wmh_mask', nifti_gz_format,
                    description="Manual WMH segmentations"),
        DatasetSpec('t2_coreg', nifti_gz_format, t2_registration_pipeline,
                    description="T2 registered to T1 weighted"),
        DatasetSpec('t1_masked', nifti_gz_format, t1_brain_mask_pipeline,
                    description="T1 masked by brain mask"),
        DatasetSpec('t2_masked', nifti_gz_format, t2_brain_mask_pipeline,
                    description="Coregistered T2 masked by brain mask"),
        DatasetSpec('brain_mask', nifti_gz_format, t2_brain_mask_pipeline,
                    description="Brain mask generated from coregistered T2"),
        DatasetSpec('manual_wmh_mask_coreg', nifti_gz_format,
                    manual_wmh_mask_registration_pipeline,
                    description="Manual WMH segmentations coregistered to T1"),
        DatasetSpec('t2_coreg_matrix', text_matrix_format,
                    t2_registration_pipeline,
                    description="Coregistration matrix for T2 to T1"),
        DatasetSpec('fs_recon_all', freesurfer_recon_all_format,
                    freesurfer_pipeline,
                    description="Output directory from Freesurfer recon_all"))
