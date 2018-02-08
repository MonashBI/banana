from itertools import chain
from copy import copy
from nipype.interfaces.freesurfer.preprocess import ReconAll
# from nianalysis.interfaces.utils import DummyReconAll as ReconAll
from nianalysis.requirements import freesurfer_req
from nianalysis.citations import (
    freesurfer_cites, optimal_t1_bet_params_cite)
from nianalysis.data_formats import (
    freesurfer_recon_all_format, nifti_gz_format, text_matrix_format,
    directory_format)
from nianalysis.dataset import DatasetSpec
from nianalysis.interfaces.utils import ZipDir, JoinPath
from ..base import MRIStudy
from nianalysis.citations import fsl_cite
from nianalysis.study.base import set_data_specs
from ..coregistered import CoregisteredStudy
from ...combined import CombinedStudy
from nianalysis.interfaces.custom.motion_correction import (
    MotionMatCalculation)


class T1Study(MRIStudy):

    def brain_mask_pipeline(self, robust=True, f_threshold=0.55,
                            g_threshold=-0.1, **kwargs):
        pipeline = super(T1Study, self).brain_mask_pipeline(
            robust=robust, f_threshold=f_threshold,
            g_threshold=g_threshold, **kwargs)
        pipeline.citations.append(optimal_t1_bet_params_cite)
        return pipeline

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
        # Zip directory before returning
        zip_dir = pipeline.create_node(interface=ZipDir(), name='zip_dir')
        zip_dir.inputs.extension = '.fs'
        pipeline.connect(join, 'path', zip_dir, 'dirname')
        # Connect inputs/outputs
        pipeline.connect_input('primary', recon_all, 'T1_files')
        pipeline.connect_output('fs_recon_all', zip_dir, 'zipped')
        pipeline.assert_connected()
        return pipeline

    _data_specs = set_data_specs(
        DatasetSpec('fs_recon_all', freesurfer_recon_all_format,
                    freesurfer_pipeline),
        inherit_from=chain(MRIStudy.data_specs()))


class CoregisteredT1Study(CombinedStudy):

    sub_study_specs = {
        't1': (T1Study, {
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
        't1', T1Study.basic_preproc_pipeline)

    t1_bet_pipeline = CombinedStudy.translate(
        't1', T1Study.brain_mask_pipeline)

    ref_bet_pipeline = CombinedStudy.translate(
        'reference', MRIStudy.brain_mask_pipeline)

    ref_basic_preproc_pipeline = CombinedStudy.translate(
        'reference', MRIStudy.basic_preproc_pipeline,
        override_default_options={'resolution': [1]})

    t1_qform_transform_pipeline = CombinedStudy.translate(
        'coreg', CoregisteredStudy.qform_transform_pipeline)

    t1_brain_mask_pipeline = CombinedStudy.translate(
        't1', T1Study.brain_mask_pipeline)

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
