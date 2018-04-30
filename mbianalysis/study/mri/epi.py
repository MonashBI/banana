
from base import MRIStudy
from nianalysis.dataset import DatasetSpec, FieldSpec
from nianalysis.data_formats import (nifti_gz_format, text_matrix_format,
                                     text_format, directory_format, par_format,
                                     dicom_format)
from nianalysis.citations import fsl_cite
from nipype.interfaces import fsl
from nianalysis.requirements import fsl509_req
from nianalysis.study.base import StudyMetaClass
from .coregistered import CoregisteredStudy
from nianalysis.study.multi import (
    MultiStudy, SubStudySpec, MultiStudyMetaClass)
from mbianalysis.interfaces.custom.motion_correction import (
    MotionMatCalculation, MergeListMotionMat)
from nianalysis.options import OptionSpec


class EPIStudy(MRIStudy):

    __metaclass__ = StudyMetaClass

    add_data_specs = [
        DatasetSpec('moco', nifti_gz_format,
                    'motion_alignment_pipeline'),
        DatasetSpec('moco_mat', directory_format,
                    'motion_alignment_pipeline'),
        DatasetSpec('moco_par', text_format,
                    'motion_alignment_pipeline')]

    add_option_specs = [
        OptionSpec('bet_robust', True),
        OptionSpec('bet_f_threshold', 0.2),
        OptionSpec('bet_reduce_bias', False)]

    def header_info_extraction_pipeline(self, **kwargs):
        return (super(EPIStudy, self).
                header_info_extraction_pipeline_factory(
                    'primary', **kwargs))

    def motion_alignment_pipeline(self, **kwargs):

        pipeline = self.create_pipeline(
            name='MCFLIRT_pipeline',
            inputs=[DatasetSpec('preproc', nifti_gz_format)],
            outputs=[DatasetSpec('moco', nifti_gz_format),
                     DatasetSpec('moco_mat', directory_format),
                     DatasetSpec('moco_par', par_format)],
            description=("Intra-epi volumes alignment."),
            version=1,
            citations=[fsl_cite],
            **kwargs)
        mcflirt = pipeline.create_node(fsl.MCFLIRT(), name='mcflirt',
                                       requirements=[fsl509_req])
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

        return pipeline


class CoregisteredEPIStudy(MultiStudy):

    __metaclass__ = MultiStudyMetaClass

    add_sub_study_specs = [
        SubStudySpec('epi', EPIStudy, {
            'epi': 'primary',
            'epi_nifti': 'primary_nifti',
            'epi_preproc': 'preproc',
            'epi_brain': 'masked',
            'epi_brain_mask': 'brain_mask',
            'epi_moco': 'moco',
            'epi_moco_mat': 'moco_mat',
            'epi_moco_par': 'moco_par',
            'ped': 'ped',
            'pe_angle': 'pe_angle',
            'tr': 'tr',
            'real_duration': 'real_duration',
            'tot_duration': 'tot_duration',
            'start_time': 'start_time',
            'dcm_info': 'dcm_info'}),
        SubStudySpec('reference', MRIStudy, {
            'reference': 'primary_nifti',
            'ref_preproc': 'preproc',
            'ref_brain': 'masked',
            'ref_brain_mask': 'brain_mask',
            'ref_wmseg': 'wm_seg'}),
        SubStudySpec('coreg', CoregisteredStudy, {
            'epi_brain': 'to_register',
            'ref_brain': 'reference',
            'epi_qformed': 'qformed',
            'epi_qform_mat': 'qform_mat'})]

    add_data_specs = [
        DatasetSpec('epi', dicom_format),
        DatasetSpec('reference', nifti_gz_format),
        DatasetSpec('epi_preproc', nifti_gz_format,
                    'epi_basic_preproc_pipeline'),
        DatasetSpec('epi_nifti', nifti_gz_format,
                    'epi_dcm2nii_pipeline'),
        DatasetSpec('epi_brain', nifti_gz_format,
                    'epi_brain_mask_pipeline'),
        DatasetSpec('epi_brain_mask', nifti_gz_format,
                    'epi_brain_mask_pipeline'),
        DatasetSpec('ref_preproc', nifti_gz_format,
                    'ref_basic_preproc_pipeline'),
        DatasetSpec('epi_qformed', nifti_gz_format,
                    'epi_qform_transform_pipeline'),
        DatasetSpec('masked', nifti_gz_format,
                    'epi_bet_pipeline'),
        DatasetSpec('epi_qform_mat', text_matrix_format,
                    'epi_qform_transform_pipeline'),
        DatasetSpec('ref_brain', nifti_gz_format, 'ref_bet_pipeline'),
        DatasetSpec('ref_brain_mask', nifti_gz_format,
                    'ref_bet_pipeline'),
        DatasetSpec('ref_wmseg', nifti_gz_format, 'ref_segmentation_pipeline'),
        DatasetSpec('epi_epireg', nifti_gz_format, 'epireg_pipeline'),
        DatasetSpec('epi_epireg_mat', text_matrix_format,
                    'epireg_pipeline'),
        DatasetSpec('motion_mats', directory_format,
                    'motion_mat_pipeline'),
        DatasetSpec('epi_moco', nifti_gz_format,
                    'epi_motion_alignment_pipeline'),
        DatasetSpec('epi_moco_mat', directory_format,
                    'epi_motion_alignment_pipeline'),
        DatasetSpec('epi_moco_par', par_format,
                    'epi_motion_alignment_pipeline'),
        DatasetSpec('dcm_info', text_format, 'epi_dcm_info_pipeline'),
        FieldSpec('ped', str, 'epi_dcm_info_pipeline'),
        FieldSpec('pe_angle', str, 'epi_dcm_info_pipeline'),
        FieldSpec('tr', float, 'epi_dcm_info_pipeline'),
        FieldSpec('start_time', str, 'epi_dcm_info_pipeline'),
        FieldSpec('real_duration', str, 'epi_dcm_info_pipeline'),
        FieldSpec('tot_duration', str, 'epi_dcm_info_pipeline')]

    add_option_specs = [OptionSpec('reference_resolution', [1]),
                        OptionSpec('multivol', True)]

    epi_basic_preproc_pipeline = MultiStudy.translate(
        'epi', 'basic_preproc_pipeline')

    epi_dcm2nii_pipeline = MultiStudy.translate(
        'epi', 'dcm2nii_conversion_pipeline')

    epi_bet_pipeline = MultiStudy.translate(
        'epi', 'brain_mask_pipeline')

    epi_dcm_info_pipeline = MultiStudy.translate(
        'epi', 'header_info_extraction_pipeline')

    ref_bet_pipeline = MultiStudy.translate(
        'reference', 'brain_mask_pipeline')

    ref_dcm2nii_pipeline = MultiStudy.translate(
        'reference', 'dcm2nii_conversion_pipeline')

    ref_segmentation_pipeline = MultiStudy.translate(
        'reference', 'segmentation_pipeline')

    ref_basic_preproc_pipeline = MultiStudy.translate(
        'reference', 'basic_preproc_pipeline')

    epi_qform_transform_pipeline = MultiStudy.translate(
        'coreg', 'qform_transform_pipeline')

    epi_motion_alignment_pipeline = MultiStudy.translate(
        'epi', 'motion_alignment_pipeline')

    epi_brain_mask_pipeline = MultiStudy.translate(
        'epi', 'brain_mask_pipeline')

    def epireg_pipeline(self, **kwargs):

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
            version=1,
            citations=[fsl_cite],
            **kwargs)
        epireg = pipeline.create_node(fsl.epi.EpiReg(), name='epireg',
                                      requirements=[fsl509_req])

        epireg.inputs.out_base = 'epireg2ref'
        pipeline.connect_input('epi_brain', epireg, 'epi')
        pipeline.connect_input('ref_brain', epireg, 't1_brain')
        pipeline.connect_input('ref_preproc', epireg, 't1_head')
        pipeline.connect_input('ref_wmseg', epireg, 'wmseg')

        pipeline.connect_output('epi_epireg', epireg, 'out_file')
        pipeline.connect_output('epi_epireg_mat', epireg, 'epi2str_mat')
        return pipeline

    def motion_mat_pipeline(self, **kwargs):

        pipeline = self.create_pipeline(
            name='motion_mat_calculation',
            inputs=[DatasetSpec('epi_epireg_mat', text_matrix_format),
                    DatasetSpec('epi_qform_mat', text_matrix_format),
                    DatasetSpec('epi_moco_mat', directory_format)],
            outputs=[DatasetSpec('motion_mats', directory_format)],
            description=("EPI Motion matrices calculation"),
            version=1,
            citations=[fsl_cite],
            **kwargs)

        mm = pipeline.create_node(
            MotionMatCalculation(), name='motion_mats')
        pipeline.connect_input('epi_epireg_mat', mm, 'reg_mat')
        pipeline.connect_input('epi_qform_mat', mm, 'qform_mat')
        pipeline.connect_input('epi_moco_mat', mm, 'align_mats')
        pipeline.connect_output('motion_mats', mm, 'motion_mats')
        return pipeline
