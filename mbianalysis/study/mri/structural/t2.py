from nianalysis.study.base import set_data_specs
from nianalysis.dataset import DatasetSpec, FieldSpec
from nianalysis.data_formats import nifti_gz_format
from nianalysis.data_formats import (text_matrix_format, directory_format,
                                     text_format)
from ..base import MRIStudy
from nianalysis.citations import fsl_cite
from ..coregistered import CoregisteredStudy
from nianalysis.study.combined import CombinedStudy
from mbianalysis.interfaces.custom.motion_correction import (
    MotionMatCalculation)


class T2Study(MRIStudy):

    def brain_mask_pipeline(self, robust=True, f_threshold=0.5,
                            reduce_bias=False, **kwargs):
        return super(T2Study, self).brain_mask_pipeline(
            robust=robust, f_threshold=f_threshold, reduce_bias=reduce_bias,
            **kwargs)

    def header_info_extraction_pipeline(self, **kwargs):
        return (super(T2Study, self).
                header_info_extraction_pipeline_factory(
                    'primary', **kwargs))

    _data_specs = set_data_specs(
        inherit_from=MRIStudy.data_specs())


class CoregisteredT2Study(CombinedStudy):

    sub_study_specs = {
        't2': (T2Study, {
            't2': 'primary',
            't2_nifti': 'primary_nifti',
            't2_preproc': 'preproc',
            't2_brain': 'masked',
            't2_brain_mask': 'brain_mask',
            't2_ped': 'ped',
            't2_pe_angle': 'pe_angle',
            't2_tr': 'tr',
            't2_real_duration': 'real_duration',
            't2_tot_duration': 'tot_duration',
            't2_start_time': 'start_time',
            't2_dcm_info': 'dcm_info'}),
        'reference': (MRIStudy, {
            'reference': 'primary_nifti',
            'ref_preproc': 'preproc',
            'ref_brain': 'masked',
            'ref_brain_mask': 'brain_mask'}),
        'coreg': (CoregisteredStudy, {
            't2_brain': 'to_register',
            'ref_brain': 'reference',
            't2_qformed': 'qformed',
            't2_qform_mat': 'qform_mat',
            't2_reg': 'registered',
            't2_reg_mat': 'matrix'})}

    t2_basic_preproc_pipeline = CombinedStudy.translate(
        't2', T2Study.basic_preproc_pipeline)

    t2_dcm2nii_pipeline = CombinedStudy.translate(
        't2', MRIStudy.dcm2nii_conversion_pipeline)

    t2_dcm_info_pipeline = CombinedStudy.translate(
        't2', T2Study.header_info_extraction_pipeline,
        override_default_options={'multivol': False})

    t2_bet_pipeline = CombinedStudy.translate(
        't2', T2Study.brain_mask_pipeline)

    ref_bet_pipeline = CombinedStudy.translate(
        'reference', MRIStudy.brain_mask_pipeline)

    ref_basic_preproc_pipeline = CombinedStudy.translate(
        'reference', MRIStudy.basic_preproc_pipeline,
        override_default_options={'resolution': [1]})

    t2_qform_transform_pipeline = CombinedStudy.translate(
        'coreg', CoregisteredStudy.qform_transform_pipeline)

    t2_brain_mask_pipeline = CombinedStudy.translate(
        't2', T2Study.brain_mask_pipeline)

    t2_rigid_registration_pipeline = CombinedStudy.translate(
        'coreg', CoregisteredStudy.linear_registration_pipeline)

    def t2_motion_mat_pipeline(self, **options):

        pipeline = self.create_pipeline(
            name='t2_motion_mat_calculation',
            inputs=[DatasetSpec('t2_reg_mat', text_matrix_format),
                    DatasetSpec('t2_qform_mat', text_matrix_format)],
            outputs=[DatasetSpec('t2_motion_mats', directory_format)],
            description=("T2w Motion matrices calculation"),
            default_options={},
            version=1,
            citations=[fsl_cite],
            options=options)

        mm = pipeline.create_node(
            MotionMatCalculation(), name='t2_motion_mats')
        pipeline.connect_input('t2_reg_mat', mm, 'reg_mat')
        pipeline.connect_input('t2_qform_mat', mm, 'qform_mat')
        pipeline.connect_output('t2_motion_mats', mm, 'motion_mats')
        pipeline.assert_connected()
        return pipeline

    _data_specs = set_data_specs(
        DatasetSpec('t2', nifti_gz_format),
        DatasetSpec('t2_nifti', nifti_gz_format, t2_dcm2nii_pipeline),
        DatasetSpec('reference', nifti_gz_format),
        DatasetSpec('t2_preproc', nifti_gz_format, t2_basic_preproc_pipeline),
        DatasetSpec('t2_brain', nifti_gz_format, t2_brain_mask_pipeline),
        DatasetSpec('t2_brain_mask', nifti_gz_format, t2_brain_mask_pipeline),
        DatasetSpec('ref_preproc', nifti_gz_format,
                    ref_basic_preproc_pipeline),
        DatasetSpec('t2_qformed', nifti_gz_format,
                    t2_qform_transform_pipeline),
        DatasetSpec('masked', nifti_gz_format, t2_bet_pipeline),
        DatasetSpec('t2_qform_mat', text_matrix_format,
                    t2_qform_transform_pipeline),
        DatasetSpec('ref_brain', nifti_gz_format, ref_bet_pipeline),
        DatasetSpec('ref_brain_mask', nifti_gz_format,
                    ref_bet_pipeline),
        DatasetSpec('t2_reg', nifti_gz_format, t2_rigid_registration_pipeline),
        DatasetSpec('t2_reg_mat', text_matrix_format,
                    t2_rigid_registration_pipeline),
        DatasetSpec('t2_motion_mats', directory_format,
                    t2_motion_mat_pipeline),
        DatasetSpec('t2_dcm_info', text_format, t2_dcm_info_pipeline),
        FieldSpec('t2_ped', str, t2_dcm_info_pipeline),
        FieldSpec('t2_pe_angle', str, t2_dcm_info_pipeline),
        FieldSpec('t2_tr', float, t2_dcm_info_pipeline),
        FieldSpec('t2_start_time', str, t2_dcm_info_pipeline),
        FieldSpec('t2_real_duration', str, t2_dcm_info_pipeline),
        FieldSpec('t2_tot_duration', str, t2_dcm_info_pipeline))
