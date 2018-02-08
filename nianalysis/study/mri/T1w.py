from base import MRIStudy
from nianalysis.dataset import DatasetSpec
from nianalysis.data_formats import (nifti_gz_format, text_matrix_format,
                                     directory_format)
from nianalysis.citations import fsl_cite
from nianalysis.study.base import set_data_specs
from .coregistered import CoregisteredStudy
from ..combined import CombinedStudy
from nianalysis.interfaces.custom.motion_correction import (
    MotionMatCalculation)


class T1WStudy(MRIStudy):

    def brain_mask_pipeline(self, robust=True, f_threshold=0.55,
                            g_threshold=-0.1, **kwargs):
        return super(T1WStudy, self).brain_mask_pipeline(
            robust=robust, f_threshold=f_threshold, g_threshold=g_threshold,
            **kwargs)

    _data_specs = set_data_specs(inherit_from=MRIStudy.data_specs())


class CoregisteredT1WStudy(CombinedStudy):

    sub_study_specs = {
        't1': (T1WStudy, {
            't1': 'primary',
            't1_preproc': 'preproc',
            't1_brain': 'masked',
            't1_brain_mask': 'brain_mask'}),
        'reference': (MRIStudy, {
            'reference': 'primary',
            'ref_preproc': 'preproc',
            'ref_brain': 'masked',
            'ref_brain_mask': 'brain_mask'}),
        'coreg': (CoregisteredStudy, {
            't1_brain': 'to_register',
            'ref_brain': 'reference',
            't1_qformed': 'qformed',
            't1_qform_mat': 'qform_mat',
            't1_reg': 'registered',
            't1_reg_mat': 'matrix'})}

    t1_basic_preproc_pipeline = CombinedStudy.translate(
        't1', T1WStudy.basic_preproc_pipeline)

    t1_bet_pipeline = CombinedStudy.translate(
        't1', T1WStudy.brain_mask_pipeline)

    ref_bet_pipeline = CombinedStudy.translate(
        'reference', MRIStudy.brain_mask_pipeline)

    ref_basic_preproc_pipeline = CombinedStudy.translate(
        'reference', MRIStudy.basic_preproc_pipeline,
        override_default_options={'resolution': [1]})

    t1_qform_transform_pipeline = CombinedStudy.translate(
        'coreg', CoregisteredStudy.qform_transform_pipeline)

    t1_brain_mask_pipeline = CombinedStudy.translate(
        't1', T1WStudy.brain_mask_pipeline)

    t1_rigid_registration_pipeline = CombinedStudy.translate(
        'coreg', CoregisteredStudy.linear_registration_pipeline)

    def t1_motion_mat_pipeline(self, **options):

        pipeline = self.create_pipeline(
            name='t1_motion_mat_calculation',
            inputs=[DatasetSpec('t1_reg_mat', text_matrix_format),
                    DatasetSpec('t1_qform_mat', text_matrix_format)],
            outputs=[DatasetSpec('t1_motion_mats', directory_format)],
            description=("T1w Motion matrices calculation"),
            default_options={},
            version=1,
            citations=[fsl_cite],
            options=options)

        mm = pipeline.create_node(
            MotionMatCalculation(), name='t1_motion_mats')
        pipeline.connect_input('t1_reg_mat', mm, 'reg_mat')
        pipeline.connect_input('t1_qform_mat', mm, 'qform_mat')
        pipeline.connect_output('t1_motion_mats', mm, 'motion_mats')
        pipeline.assert_connected()
        return pipeline

    _data_specs = set_data_specs(
        DatasetSpec('t1', nifti_gz_format),
        DatasetSpec('reference', nifti_gz_format),
        DatasetSpec('t1_preproc', nifti_gz_format, t1_basic_preproc_pipeline),
        DatasetSpec('t1_brain', nifti_gz_format, t1_brain_mask_pipeline),
        DatasetSpec('t1_brain_mask', nifti_gz_format, t1_brain_mask_pipeline),
        DatasetSpec('ref_preproc', nifti_gz_format,
                    ref_basic_preproc_pipeline),
        DatasetSpec('t1_qformed', nifti_gz_format,
                    t1_qform_transform_pipeline),
        DatasetSpec('masked', nifti_gz_format, t1_bet_pipeline),
        DatasetSpec('t1_qform_mat', text_matrix_format,
                    t1_qform_transform_pipeline),
        DatasetSpec('ref_brain', nifti_gz_format, ref_bet_pipeline),
        DatasetSpec('ref_brain_mask', nifti_gz_format,
                    ref_bet_pipeline),
        DatasetSpec('t1_reg', nifti_gz_format, t1_rigid_registration_pipeline),
        DatasetSpec('t1_reg_mat', text_matrix_format,
                    t1_rigid_registration_pipeline),
        DatasetSpec('t1_motion_mats', directory_format,
                    t1_motion_mat_pipeline))
