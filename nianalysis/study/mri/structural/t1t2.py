from nipype.interfaces.fsl import ApplyMask
from nianalysis.file_format import (
    nifti_gz_format, freesurfer_recon_all_format, text_matrix_format)
from arcana.data import FilesetSpec
from arcana.study.multi import (
    MultiStudy, SubStudySpec, MultiStudyMetaClass)
from ..coregistered import CoregisteredStudy, CoregisteredToMatrixStudy
from .t1 import T1Study
from .t2 import T2Study
from nianalysis.requirement import fsl5_req
from nianalysis.citation import fsl_cite


class T1T2Study(MultiStudy, metaclass=MultiStudyMetaClass):
    """
    T1 and T2 weighted MR fileset, with the T2-weighted coregistered to the T1.
    """

    add_sub_study_specs = [
        SubStudySpec('t1', T1Study, {
            't1': 'magnitude',
            't1_coreg_to_atlas': 'coreg_to_atlas',
            'coreg_to_atlas_coeff': 'coreg_to_atlas_coeff',
            'brain_mask': 'brain_mask',
            't1_brain': 'brain',
            'fs_recon_all': 'fs_recon_all'}),
        SubStudySpec('t2', T2Study, {
            't2_coreg': 'magnitude',
            'manual_wmh_mask_coreg': 'manual_wmh_mask',
            't2_brain': 'brain',
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
            'manual_wmh_mask_coreg': 'registered'})]

    add_data_specs = [
        FilesetSpec('t1', nifti_gz_format,
                    desc="Raw T1-weighted image (e.g. MPRAGE)"),
        FilesetSpec('t2', nifti_gz_format,
                    desc="Raw T2-weighted image (e.g. FLAIR)"),
        FilesetSpec('manual_wmh_mask', nifti_gz_format,
                    desc="Manual WMH segmentations"),
        FilesetSpec('t2_coreg', nifti_gz_format, 't2_registration_pipeline',
                    desc="T2 registered to T1 weighted"),
        FilesetSpec('t1_brain', nifti_gz_format, 't1_brain_extraction_pipeline',
                    desc="T1 brain by brain mask"),
        FilesetSpec('t2_brain', nifti_gz_format, 't2_brain_extraction_pipeline',
                    desc="Coregistered T2 brain by brain mask"),
        FilesetSpec('brain_mask', nifti_gz_format, 't2_brain_extraction_pipeline',
                    desc="Brain mask generated from coregistered T2"),
        FilesetSpec('manual_wmh_mask_coreg', nifti_gz_format,
                    'manual_wmh_mask_registration_pipeline',
                    desc="Manual WMH segmentations coregistered to T1"),
        FilesetSpec('t2_coreg_matrix', text_matrix_format,
                    't2_registration_pipeline',
                    desc="Coregistration matrix for T2 to T1"),
        FilesetSpec('t1_coreg_to_atlas', nifti_gz_format,
                    'coregister_to_atlas_pipeline'),
        FilesetSpec('coreg_to_atlas_coeff', nifti_gz_format,
                    'coregister_to_atlas_pipeline'),
        FilesetSpec('fs_recon_all', freesurfer_recon_all_format,
                    'freesurfer_pipeline',
                    desc="Output directory from Freesurfer recon_all")]

    def freesurfer_pipeline(self, **kwargs):
        pipeline = self.TranslatedPipeline(
            self, self.t1, T1Study.freesurfer_pipeline,
            add_inputs=[FilesetSpec('t2_coreg', nifti_gz_format)],
            **kwargs)
        recon_all = pipeline.node('recon_all')
        # Connect T2-weighted input
        pipeline.connect_input('t2_coreg', recon_all, 'T2_file')
        recon_all.inputs.use_T2 = True
        return pipeline

    coregister_to_atlas_pipeline = MultiStudy.translate(
        't1', 'coregister_to_atlas_pipeline')

    t2_registration_pipeline = MultiStudy.translate(
        't2coregt1', 'linear_registration_pipeline')

    manual_wmh_mask_registration_pipeline = MultiStudy.translate(
        'wmhcoregt1',
        'linear_registration_pipeline')

    t2_brain_extraction_pipeline = MultiStudy.translate(
        't2', 'brain_extraction_pipeline')

    def t1_brain_extraction_pipeline(self, **kwargs):
        """
        Masks the T1 image using the coregistered T2 brain mask as the brain
        mask from T2 is usually more reliable (using BET in any case)
        """
        pipeline = self.pipeline(
            name='t1_brain_extraction_pipeline',
            inputs=[FilesetSpec('t1', nifti_gz_format),
                    FilesetSpec('brain_mask', nifti_gz_format)],
            outputs=[FilesetSpec('t1_brain', nifti_gz_format)],
            desc="Mask T1 with T2 brain mask",
            citations=[fsl_cite],
            **kwargs)
        # Create apply mask node
        apply_mask = pipeline.create_node(
            ApplyMask(), name='appy_mask', requirements=[fsl5_req])
        apply_mask.inputs.output_type = 'NIFTI_GZ'
        # Connect inputs
        pipeline.connect_input('t1', apply_mask, 'in_file')
        pipeline.connect_input('brain_mask', apply_mask, 'mask_file')
        # Connect outputs
        pipeline.connect_output('t1_brain', apply_mask, 'out_file')
        # Check and return
        return pipeline
