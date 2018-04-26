from nipype.interfaces.fsl import ApplyMask
from nianalysis.data_formats import (
    nifti_gz_format, freesurfer_recon_all_format, text_matrix_format)
from nianalysis.study.base import set_specs
from nianalysis.dataset import DatasetSpec
from nianalysis.study.multi import (
    MultiStudy, translate_pipeline, SubStudySpec, MultiStudyMetaClass)
from ..coregistered import CoregisteredStudy, CoregisteredToMatrixStudy
from .t1 import T1Study
from .t2 import T2Study
from nianalysis.requirements import fsl5_req
from nianalysis.citations import fsl_cite


class T1T2Study(MultiStudy):

    __metaclass__ = MultiStudyMetaClass
    """
    T1 and T2 weighted MR dataset, with the T2-weighted coregistered to the T1.
    """

    def freesurfer_pipeline(self, **options):
        pipeline = self.TranslatedPipeline(
            self, self.t1, T1Study.freesurfer_pipeline, options,
            default_options={'use_T2': True},
            add_inputs=[DatasetSpec('t2_coreg', nifti_gz_format)])
        recon_all = pipeline.node('recon_all')
        # Connect T2-weighted input
        pipeline.connect_input('t2_coreg', recon_all, 'T2_file')
        pipeline.assert_connected()
        return pipeline

    coregister_to_atlas_pipeline = translate_pipeline(
        't1', T1Study.coregister_to_atlas_pipeline)

    t2_registration_pipeline = translate_pipeline(
        't2coregt1', CoregisteredStudy.linear_registration_pipeline)

    manual_wmh_mask_registration_pipeline = translate_pipeline(
        'wmhcoregt1',
        CoregisteredToMatrixStudy.linear_registration_pipeline)

    t2_brain_mask_pipeline = translate_pipeline(
        't2', T2Study.brain_mask_pipeline)

    def t1_brain_mask_pipeline(self, **options):
        """
        Masks the T1 image using the coregistered T2 brain mask as the brain
        mask from T2 is usually more reliable (using BET in any case)
        """
        pipeline = self.create_pipeline(
            name='t1_brain_mask_pipeline',
            inputs=[DatasetSpec('t1', nifti_gz_format),
                    DatasetSpec('brain_mask', nifti_gz_format)],
            outputs=[DatasetSpec('t1_masked', nifti_gz_format)],
            default_options={},
            version=1,
            description="Mask T1 with T2 brain mask",
            citations=[fsl_cite],
            options=options)
        # Create apply mask node
        apply_mask = pipeline.create_node(
            ApplyMask(), name='appy_mask', requirements=[fsl5_req])
        apply_mask.inputs.output_type = 'NIFTI_GZ'
        # Connect inputs
        pipeline.connect_input('t1', apply_mask, 'in_file')
        pipeline.connect_input('brain_mask', apply_mask, 'mask_file')
        # Connect outputs
        pipeline.connect_output('t1_masked', apply_mask, 'out_file')
        # Check and return
        pipeline.assert_connected()
        return pipeline

    _sub_study_specs = set_specs(
        SubStudySpec('t1', T1Study, {
            't1': 'primary',
            't1_coreg_to_atlas': 'coreg_to_atlas',
            'coreg_to_atlas_coeff': 'coreg_to_atlas_coeff',
            'brain_mask': 'brain_mask',
            't1_masked': 'masked',
            'fs_recon_all': 'fs_recon_all'}),
        SubStudySpec('t2', T2Study, {
            't2_coreg': 'primary',
            'manual_wmh_mask_coreg': 'manual_wmh_mask',
            't2_masked': 'masked',
            'brain_mask': 'brain_mask'}),
        SubStudySpec('t2coregt1', CoregisteredStudy, {
            't1': 'reference',
            't2': 'to_register',
            't2_coreg': 'registered',
            't2_coreg_matrix': 'matrix'}),
        SubStudySpec('wmhcoregt1', CoregisteredToMatrixStudy, {
            't1': 'reference',
            'manual_wmh_mask': 'to_register',
            't2_coreg_matrix': 'matrix',
            'manual_wmh_mask_coreg': 'registered'}))

    _data_specs = set_specs(
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
        DatasetSpec('t1_coreg_to_atlas', nifti_gz_format,
                    'coregister_to_atlas_pipeline'),
        DatasetSpec('coreg_to_atlas_coeff', nifti_gz_format,
                    'coregister_to_atlas_pipeline'),
        DatasetSpec('fs_recon_all', freesurfer_recon_all_format,
                    freesurfer_pipeline,
                    description="Output directory from Freesurfer recon_all"))
