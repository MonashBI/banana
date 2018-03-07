from itertools import chain
from copy import copy
from nipype.interfaces.freesurfer.preprocess import ReconAll
# from nianalysis.interfaces.utils import DummyReconAll as ReconAll
from nianalysis.requirements import freesurfer_req
from nianalysis.citations import (
    freesurfer_cites, optimal_t1_bet_params_cite)
from nianalysis.data_formats import (
    freesurfer_recon_all_format, nifti_gz_format, text_matrix_format,
    directory_format, dicom_format, text_format)
from nianalysis.dataset import DatasetSpec, FieldSpec
from nianalysis.interfaces.utils import ZipDir, JoinPath
from ..base import MRIStudy
from nianalysis.citations import fsl_cite
from nianalysis.study.base import set_specs
from ..coregistered import CoregisteredStudy
from nianalysis.study.multi import (MultiStudy, SubStudySpec, MultiStudyMetaClass)
from mbianalysis.interfaces.custom.motion_correction import (
    MotionMatCalculation)


class T1Study(MRIStudy):

    def brain_mask_pipeline(self, robust=True, f_threshold=0.65,
                            g_threshold=-0.1, **kwargs):
        pipeline = super(T1Study, self).brain_mask_pipeline(
            robust=robust, f_threshold=f_threshold,
            g_threshold=g_threshold, **kwargs)
        pipeline.citations.append(optimal_t1_bet_params_cite)
        return pipeline

    def header_info_extraction_pipeline(self, **kwargs):
        return (super(T1Study, self).
                header_info_extraction_pipeline_factory(
                    'primary', **kwargs))

    def freesurfer_pipeline(self, num_processes=16, **options):  # @UnusedVariable @IgnorePep8
        """
        Segments grey matter, white matter and CSF from T1 images using
        SPM "NewSegment" function.

        NB: Default values come from the W2MHS toolbox
        """
        pipeline = self.create_pipeline(
            name='segmentation',
            inputs=[DatasetSpec('primary', nifti_gz_format)],
            outputs=[DatasetSpec('fs_recon_all', freesurfer_recon_all_format)],
            description="Segment white/grey matter and csf",
            default_options={},
            version=1,
            citations=copy(freesurfer_cites),
            options=options)
        # FS ReconAll node
        recon_all = pipeline.create_node(
            interface=ReconAll(), name='recon_all',
            requirements=[freesurfer_req], wall_time=2000)
        recon_all.inputs.directive = 'all'
        recon_all.inputs.openmp = num_processes
        # Wrapper around os.path.join
        join = pipeline.create_node(interface=JoinPath(), name='join')
        pipeline.connect(recon_all, 'subjects_dir', join, 'dirname')
        pipeline.connect(recon_all, 'subject_id', join, 'filename')
        # Connect inputs/outputs
        pipeline.connect_input('primary', recon_all, 'T1_files')
        pipeline.connect_output('fs_recon_all', join, 'path')
        pipeline.assert_connected()
        return pipeline

    _data_specs = set_specs(
        DatasetSpec('fs_recon_all', freesurfer_recon_all_format,
                    freesurfer_pipeline),
        inherit_from=MRIStudy.data_specs())


class CoregisteredT1Study(MultiStudy):

    __metaclass__ = MultiStudyMetaClass

    t1_basic_preproc_pipeline = MultiStudy.translate(
        't1', T1Study.basic_preproc_pipeline)

    t1_dcm2nii_pipeline = MultiStudy.translate(
        't1', MRIStudy.dcm2nii_conversion_pipeline)

    t1_dcm_info_pipeline = MultiStudy.translate(
        't1', T1Study.header_info_extraction_pipeline,
        override_default_options={'multivol': False})

    t1_bet_pipeline = MultiStudy.translate(
        't1', T1Study.brain_mask_pipeline)

    ref_bet_pipeline = MultiStudy.translate(
        'reference', MRIStudy.brain_mask_pipeline)

    ref_basic_preproc_pipeline = MultiStudy.translate(
        'reference', MRIStudy.basic_preproc_pipeline,
        override_default_options={'resolution': [1]})

    t1_qform_transform_pipeline = MultiStudy.translate(
        'coreg', CoregisteredStudy.qform_transform_pipeline)

    t1_brain_mask_pipeline = MultiStudy.translate(
        't1', T1Study.brain_mask_pipeline)

    t1_rigid_registration_pipeline = MultiStudy.translate(
        'coreg', CoregisteredStudy.linear_registration_pipeline)

    def motion_mat_pipeline(self, **options):

        pipeline = self.create_pipeline(
            name='motion_mat_calculation',
            inputs=[DatasetSpec('t1_reg_mat', text_matrix_format),
                    DatasetSpec('t1_qform_mat', text_matrix_format)],
            outputs=[DatasetSpec('motion_mats', directory_format)],
            description=("T1w Motion matrices calculation"),
            default_options={},
            version=1,
            citations=[fsl_cite],
            options=options)

        mm = pipeline.create_node(
            MotionMatCalculation(), name='motion_mats')
        pipeline.connect_input('t1_reg_mat', mm, 'reg_mat')
        pipeline.connect_input('t1_qform_mat', mm, 'qform_mat')
        pipeline.connect_output('motion_mats', mm, 'motion_mats')
        pipeline.assert_connected()
        return pipeline

    _sub_study_specs = set_specs(
        SubStudySpec('t1', T1Study, {
            't1': 'primary',
            't1_nifti': 'primary_nifti',
            't1_preproc': 'preproc',
            't1_brain': 'masked',
            't1_brain_mask': 'brain_mask',
            't1_ped': 'ped',
            't1_pe_angle': 'pe_angle',
            'tr': 'tr',
            'real_duration': 'real_duration',
            'tot_duration': 'tot_duration',
            'start_time': 'start_time',
            'dcm_info': 'dcm_info'}),
        SubStudySpec('reference', MRIStudy, {
            'reference': 'primary_nifti',
            'ref_preproc': 'preproc',
            'ref_brain': 'masked',
            'ref_brain_mask': 'brain_mask'}),
        SubStudySpec('coreg', CoregisteredStudy, {
            't1_brain': 'to_register',
            'ref_brain': 'reference',
            't1_qformed': 'qformed',
            't1_qform_mat': 'qform_mat',
            't1_reg': 'registered',
            't1_reg_mat': 'matrix'}))

    _data_specs = set_specs(
        DatasetSpec('t1', dicom_format),
        DatasetSpec('t1_nifti', nifti_gz_format, t1_dcm2nii_pipeline),
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
        DatasetSpec('motion_mats', directory_format,
                    motion_mat_pipeline),
        DatasetSpec('dcm_info', text_format, t1_dcm_info_pipeline),
        FieldSpec('ped', str, t1_dcm_info_pipeline),
        FieldSpec('pe_angle', str, t1_dcm_info_pipeline),
        FieldSpec('tr', float, t1_dcm_info_pipeline),
        FieldSpec('start_time', str, t1_dcm_info_pipeline),
        FieldSpec('real_duration', str, t1_dcm_info_pipeline),
        FieldSpec('tot_duration', str, t1_dcm_info_pipeline))
