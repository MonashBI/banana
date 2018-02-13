from base import MRIStudy
from nianalysis.dataset import DatasetSpec
from nianalysis.data_formats import (nifti_gz_format, text_matrix_format,
                                     text_format, directory_format, par_format)
from nianalysis.citations import fsl_cite
from nipype.interfaces import fsl
from nianalysis.requirements import fsl5_req
from nianalysis.study.base import set_data_specs
from .coregistered import CoregisteredStudy
from ..combined import CombinedStudy
from mbianalysis.interfaces.custom.motion_correction import (
    MotionMatCalculation, MergeListMotionMat)


class EPIStudy(MRIStudy):

    def brain_mask_pipeline(self, robust=True, f_threshold=0.2, **kwargs):
        return super(EPIStudy, self).brain_mask_pipeline(
            robust=robust, f_threshold=f_threshold, **kwargs)

    def motion_alignment_pipeline(self, **options):

        pipeline = self.create_pipeline(
            name='MCFLIRT_pipeline',
            inputs=[DatasetSpec('preproc', nifti_gz_format)],
            outputs=[DatasetSpec('moco', nifti_gz_format),
                     DatasetSpec('moco_mat', text_matrix_format),
                     DatasetSpec('moco_par', par_format)],
            description=("Intra-epi volumes alignment."),
            default_options={},
            version=1,
            citations=[fsl_cite],
            options=options)
        mcflirt = pipeline.create_node(fsl.MCFLIRT(), name='mcflirt',
                                       requirements=[fsl5_req])
        mcflirt.inputs.ref_vol = 0
        mcflirt.inputs.save_mats = True
        mcflirt.inputs.save_plots = True
        mcflirt.inputs.output_type = 'NIFTI_GZ'
        mcflirt.inputs.out_file = 'moco.nii.gz'
        pipeline.connect_input('preproc', mcflirt, 'in_file')
        pipeline.connect_output('moco', mcflirt, 'out_file')
#         pipeline.connect_output('moco_mat', mcflirt, 'mat_file')
        pipeline.connect_output('moco_par', mcflirt, 'par_file')

        merge = pipeline.create_node(MergeListMotionMat(), name='merge')
        pipeline.connect(mcflirt, 'mat_file', merge, 'file_list')
        pipeline.connect_output('moco_mat', merge, 'out_dir')

        pipeline.assert_connected()
        return pipeline

    _data_specs = set_data_specs(
        DatasetSpec('moco', nifti_gz_format, motion_alignment_pipeline),
        DatasetSpec('moco_mat', directory_format, motion_alignment_pipeline),
        DatasetSpec('moco_par', text_format, motion_alignment_pipeline),
        inherit_from=MRIStudy.data_specs())


class CoregisteredEPIStudy(CombinedStudy):

    sub_study_specs = {
        'epi': (EPIStudy, {
            'epi': 'primary',
            'epi_preproc': 'preproc',
            'epi_brain': 'masked',
            'epi_brain_mask': 'brain_mask',
            'epi_moco': 'moco',
            'epi_moco_mat': 'moco_mat',
            'epi_moco_par': 'moco_par'}),
        'reference': (MRIStudy, {
            'reference': 'primary',
            'ref_preproc': 'preproc',
            'ref_brain': 'masked',
            'ref_brain_mask': 'brain_mask',
            'ref_wmseg': 'wm_seg'}),
        'coreg': (CoregisteredStudy, {
            'epi_preproc': 'to_register',
            'ref_preproc': 'reference',
            'epi_qformed': 'qformed',
            'epi_qform_mat': 'qform_mat'})}

    epi_basic_preproc_pipeline = CombinedStudy.translate(
        'epi', EPIStudy.basic_preproc_pipeline)

    epi_bet_pipeline = CombinedStudy.translate(
        'epi', EPIStudy.brain_mask_pipeline)

    ref_bet_pipeline = CombinedStudy.translate(
        'reference', MRIStudy.brain_mask_pipeline)

    ref_segmentation_pipeline = CombinedStudy.translate(
        'reference', MRIStudy.segmentation_pipeline)

    ref_basic_preproc_pipeline = CombinedStudy.translate(
        'reference', MRIStudy.basic_preproc_pipeline,
        override_default_options={'resolution': [1]})

    epi_qform_transform_pipeline = CombinedStudy.translate(
        'coreg', CoregisteredStudy.qform_transform_pipeline)

    epi_motion_alignment_pipeline = CombinedStudy.translate(
        'epi', EPIStudy.motion_alignment_pipeline)

    epi_brain_mask_pipeline = CombinedStudy.translate(
        'epi', EPIStudy.brain_mask_pipeline)

    def epireg_pipeline(self, **options):

        pipeline = self.create_pipeline(
            name='EPIREG_pipeline',
            inputs=[DatasetSpec('epi_brain', nifti_gz_format),
                    DatasetSpec('ref_brain', nifti_gz_format),
                    DatasetSpec('ref_preproc', nifti_gz_format),
                    DatasetSpec('ref_wmseg', nifti_gz_format)],
            outputs=[DatasetSpec('epi_epireg', nifti_gz_format),
                     DatasetSpec('epi_epireg_mat', text_matrix_format)],
            description=("Intra-subjects epi registration improved "
                         "using white matter boundaries."),
            default_options={},
            version=1,
            citations=[fsl_cite],
            options=options)
        epireg = pipeline.create_node(fsl.epi.EpiReg(), name='epireg',
                                      requirements=[fsl5_req])

        epireg.inputs.out_base = 'epireg2ref'
        pipeline.connect_input('epi_brain', epireg, 'epi')
        pipeline.connect_input('ref_brain', epireg, 't1_brain')
        pipeline.connect_input('ref_preproc', epireg, 't1_head')
        pipeline.connect_input('ref_wmseg', epireg, 'wmseg')

        pipeline.connect_output('epi_epireg', epireg, 'out_file')
        pipeline.connect_output('epi_epireg_mat', epireg, 'epi2str_mat')
        pipeline.assert_connected()
        return pipeline

    def epi_motion_mat_pipeline(self, **options):

        pipeline = self.create_pipeline(
            name='epi_motion_mat_calculation',
            inputs=[DatasetSpec('epi_epireg_mat', text_matrix_format),
                    DatasetSpec('epi_qform_mat', text_matrix_format),
                    DatasetSpec('epi_moco_mat', directory_format)],
            outputs=[DatasetSpec('epi_motion_mats', directory_format)],
            description=("EPI Motion matrices calculation"),
            default_options={},
            version=1,
            citations=[fsl_cite],
            options=options)

        mm = pipeline.create_node(
            MotionMatCalculation(), name='epi_motion_mats')
        pipeline.connect_input('epi_epireg_mat', mm, 'reg_mat')
        pipeline.connect_input('epi_qform_mat', mm, 'qform_mat')
        pipeline.connect_input('epi_moco_mat', mm, 'align_mats')
        pipeline.connect_output('epi_motion_mats', mm, 'motion_mats')
        pipeline.assert_connected()
        return pipeline

    _data_specs = set_data_specs(
        DatasetSpec('epi', nifti_gz_format),
        DatasetSpec('reference', nifti_gz_format),
        DatasetSpec('epi_preproc', nifti_gz_format,
                    epi_basic_preproc_pipeline),
        DatasetSpec('epi_brain', nifti_gz_format,
                    epi_brain_mask_pipeline),
        DatasetSpec('epi_brain_mask', nifti_gz_format,
                    epi_brain_mask_pipeline),
        DatasetSpec('ref_preproc', nifti_gz_format,
                    ref_basic_preproc_pipeline),
        DatasetSpec('epi_qformed', nifti_gz_format,
                    epi_qform_transform_pipeline),
        DatasetSpec('masked', nifti_gz_format,
                    epi_bet_pipeline),
        DatasetSpec('epi_qform_mat', text_matrix_format,
                    epi_qform_transform_pipeline),
        DatasetSpec('ref_brain', nifti_gz_format, ref_bet_pipeline),
        DatasetSpec('ref_brain_mask', nifti_gz_format,
                    ref_bet_pipeline),
        DatasetSpec('ref_wmseg', nifti_gz_format, ref_segmentation_pipeline),
        DatasetSpec('epi_epireg', nifti_gz_format, epireg_pipeline),
        DatasetSpec('epi_epireg_mat', text_matrix_format,
                    epireg_pipeline),
        DatasetSpec('epi_motion_mats', directory_format,
                    epi_motion_mat_pipeline),
        DatasetSpec('epi_moco', nifti_gz_format,
                    epi_motion_alignment_pipeline),
        DatasetSpec('epi_moco_mat', directory_format,
                    epi_motion_alignment_pipeline),
        DatasetSpec('epi_moco_par', par_format,
                    epi_motion_alignment_pipeline))