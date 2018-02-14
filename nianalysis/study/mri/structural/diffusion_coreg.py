from ..base import MRIStudy
from nianalysis.dataset import DatasetSpec, FieldSpec
from nianalysis.data_formats import (
    nifti_gz_format, text_matrix_format, text_format, directory_format,
    par_format, dicom_format, eddy_par_format)
from nipype.interfaces.fsl import (ExtractROI, TOPUP, ApplyTOPUP)
from nianalysis.interfaces.custom.motion_correction import (
    PrepareDWI, CheckDwiNames, GenTopupConfigFiles)
from nianalysis.citations import fsl_cite
from nipype.interfaces import fsl
from nianalysis.requirements import fsl5_req
from nianalysis.study.base import set_data_specs
from ..coregistered import CoregisteredStudy
from ...combined import CombinedStudy
from nianalysis.interfaces.custom.motion_correction import (
    MotionMatCalculation, MergeListMotionMat)
from nianalysis.interfaces.converters import Dcm2niix
from nipype.interfaces.utility import Merge as merge_lists
from nianalysis.interfaces.mrtrix.preproc import DWIPreproc
from nipype.interfaces.fsl.utils import Merge as fsl_merge


class DiffusionStudy(MRIStudy):

    def brain_mask_pipeline(self, robust=True, f_threshold=0.2, **kwargs):
        return super(DiffusionStudy, self).brain_mask_pipeline(
            robust=robust, f_threshold=f_threshold, **kwargs)

    dcm_in = [DatasetSpec('dwi_main', dicom_format)]

    def header_info_extraction_pipeline(self, dcm_in, **kwargs):
        return super(DiffusionStudy, self).header_info_extraction_pipeline(
            dcm_in, **kwargs)

    def dwipreproc_pipeline(self, **options):

        pipeline = self.create_pipeline(
            name='dwipreproc_pipeline',
            inputs=[DatasetSpec('dwi_main', dicom_format),
                    DatasetSpec('dwi_ref', dicom_format),
                    FieldSpec('ped', dtype=str),
                    FieldSpec('phase_offset', dtype=str)],
            outputs=[DatasetSpec('preproc', nifti_gz_format),
                     DatasetSpec('eddy_par', eddy_par_format)],
            description=("Diffusion pre-processing pipeline"),
            default_options={},
            version=1,
            citations=[],
            options=options)

        converter1 = pipeline.create_node(Dcm2niix(), name='converter1')
        converter1.inputs.compression = 'y'
        pipeline.connect_input('dicom_dwi', converter1, 'input_dir')
        converter2 = pipeline.create_node(Dcm2niix(), name='converter2')
        converter2.inputs.compression = 'y'
        pipeline.connect_input('dicom_dwi_1', converter2, 'input_dir')
        prep_dwi = pipeline.create_node(PrepareDWI(), name='prepare_dwi')
        pipeline.connect_input('ped', prep_dwi, 'pe_dir')
        pipeline.connect_input('phase_offset', prep_dwi, 'phase_offset')
#         prep_dwi.inputs.pe_dir = 'ROW'
#         prep_dwi.inputs.phase_offset = '-1.5'
        pipeline.connect(converter1, 'converted', prep_dwi, 'dwi')
        pipeline.connect(converter2, 'converted', prep_dwi, 'dwi1')

        check_name = pipeline.create_node(CheckDwiNames(),
                                          name='check_names')
        pipeline.connect(prep_dwi, 'main', check_name, 'nifti_dwi')
        pipeline.connect_input('dicom_dwi', check_name, 'dicom_dwi')
        pipeline.connect_input('dicom_dwi_1', check_name, 'dicom_dwi1')
        roi = pipeline.create_node(ExtractROI(), name='extract_roi')
        roi.inputs.t_min = 0
        roi.inputs.t_size = 1
        pipeline.connect(prep_dwi, 'main', roi, 'in_file')

        merge_outputs = pipeline.create_node(merge_lists(2),
                                             name='merge_files')
        pipeline.connect(roi, 'roi_file', merge_outputs, 'in1')
        pipeline.connect(prep_dwi, 'secondary', merge_outputs, 'in2')
        merge = pipeline.create_node(fsl_merge(), name='fsl_merge')
        merge.inputs.dimension = 't'
        pipeline.connect(merge_outputs, 'out', merge, 'in_files')
        dwipreproc = pipeline.create_node(DWIPreproc(), name='dwipreproc')
        dwipreproc.inputs.eddy_options = '--data_is_shelled '
        dwipreproc.inputs.rpe_pair = True
        dwipreproc.inputs.no_clean_up = True
        dwipreproc.inputs.out_file_ext = '.nii.gz'
        dwipreproc.inputs.temp_dir = 'dwipreproc_tempdir'
        pipeline.connect(merge, 'merged_file', dwipreproc, 'se_epi')
        pipeline.connect(prep_dwi, 'pe', dwipreproc, 'pe_dir')
        pipeline.connect(check_name, 'main', dwipreproc, 'in_file')

        pipeline.connect_output('primary', dwipreproc, 'out_file')
        pipeline.connect_output('eddy_par', dwipreproc, 'eddy_parameters')

        pipeline.assert_connected()
        return pipeline

    def topup_factory(self, name, to_be_corrected_name, ref_input_name,
                      output_name, **options):

        pipeline = self.create_pipeline(
            name=name,
            inputs=[DatasetSpec(to_be_corrected_name, nifti_gz_format),
                    DatasetSpec(ref_input_name, nifti_gz_format),
                    ],
            outputs=[DatasetSpec(output_name, nifti_gz_format)],
            description=("Dimensions swapping to ensure that all the images "
                         "have the same orientations."),
            default_options={},
            version=1,
            citations=[],
            options=options)

        prep_dwi = pipeline.create_node(PrepareDWI(), name='prepare_dwi')
        prep_dwi.inputs.pe_dir = 'ROW'
        prep_dwi.inputs.phase_offset = '-1.5'
        pipeline.connect_input(to_be_corrected_name, prep_dwi, 'dwi')
        pipeline.connect_input(ref_input_name, prep_dwi, 'dwi1')
        ped1 = pipeline.create_node(GenTopupConfigFiles(), name='gen_config1')
        pipeline.connect(prep_dwi, 'pe', ped1, 'ped')
        merge_outputs1 = pipeline.create_node(merge_lists(2),
                                              name='merge_files1')
        pipeline.connect_input(to_be_corrected_name, merge_outputs1, 'in1')
        pipeline.connect_input(ref_input_name, merge_outputs1, 'in2')
        merge1 = pipeline.create_node(fsl_merge(), name='fsl_merge1')
        merge1.inputs.dimension = 't'
        pipeline.connect(merge_outputs1, 'out', merge1, 'in_files')
        topup1 = pipeline.create_node(TOPUP(), name='topup1')
        pipeline.connect(merge1, 'merged_file', topup1, 'in_file')
        pipeline.connect(ped1, 'config_file', topup1, 'encoding_file')
        in_apply_tp1 = pipeline.create_node(merge_lists(1),
                                            name='in_apply_tp1')
        pipeline.connect_input(to_be_corrected_name, in_apply_tp1, 'in1')
        apply_topup1 = pipeline.create_node(ApplyTOPUP(), name='applytopup1')
        apply_topup1.inputs.method = 'jac'
        apply_topup1.inputs.in_index = [1]
        pipeline.connect(in_apply_tp1, 'out', apply_topup1, 'in_files')
        pipeline.connect(
            ped1, 'apply_topup_config', apply_topup1, 'encoding_file')
        pipeline.connect(topup1, 'out_movpar', apply_topup1, 'in_topup_movpar')
        pipeline.connect(
            topup1, 'out_fieldcoef', apply_topup1, 'in_topup_fieldcoef')

        pipeline.connect_output(output_name, apply_topup1, 'out_corrected')
        pipeline.assert_connected()
        return pipeline

    def topup_pipeline(self, **options):
        return self.topup_factory('dwi_topup', 'topup_in', 'topup_ref',
                                  'dwi_distorted')

    _data_specs = set_data_specs(
        DatasetSpec('dwi_main', dicom_format),
        DatasetSpec('dwi_ref', dicom_format),
        DatasetSpec('topup_in', nifti_gz_format),
        DatasetSpec('topup_ref', nifti_gz_format),
        DatasetSpec('preproc', nifti_gz_format, dwipreproc_pipeline),
        DatasetSpec('eddy_par', eddy_par_format, topup_pipeline),
        inherit_from=MRIStudy.data_specs())


class DiffusioReferenceStudy(DiffusionStudy):

    dcm_in = [DatasetSpec('to_be_corrected', dicom_format)]

    def header_info_extraction_pipeline(self, dcm_in, **kwargs):
        return super(DiffusionStudy, self).header_info_extraction_pipeline(
            dcm_in, **kwargs)

    def conversion_pipeline(self, **options):
        pipeline = self.create_pipeline(
            name='dicom2nifti_coversion',
            inputs=[DatasetSpec('to_be_corrected', dicom_format)],
            outputs=[DatasetSpec('to_be_corrected_nifti', nifti_gz_format)],
            description=("Dimensions swapping to ensure that all the images "
                         "have the same orientations."),
            default_options={},
            version=1,
            citations=[],
            options=options)

        converter = pipeline.create_node(Dcm2niix(), name='converter1')
        converter.inputs.compression = 'y'
        pipeline.connect_input('to_be_corrected', converter, 'input_dir')
        pipeline.connect_output(
            'to_be_corrected_nifti', converter, 'converted')

        pipeline.assert_connected()
        return pipeline

    inputs = ['to_be_corrected_nifti', 'topup_ref', 'dwi_distorted']

    def topup(self, inputs, **kwargs):
        return super(DiffusioReferenceStudy, self).topup_pipeline(
            inputs[0], inputs[1], inputs[3], **kwargs)


class CoregisteredDWIStudy(CombinedStudy):

    sub_study_specs = {
        'dwi': (DiffusionStudy, {
            'dwi': 'dicom_dwi',
            'dwi_1': 'dicom_dwi_1',
            'dwi_brain_mask': 'brain_mask',
            'dwi_brain': 'masked',
            'dwi_preproc': 'primary',
            'dwi_eddy_par': 'eddy_par'}),
        'reference': (MRIStudy, {
            'reference': 'primary',
            'ref_preproc': 'preproc',
            'ref_brain': 'masked',
            'ref_brain_mask': 'brain_mask'}),
        'coreg': (CoregisteredStudy, {
            'dwi_brain': 'to_register',
            'ref_preproc': 'reference',
            'dwi_qformed': 'qformed',
            'dwi_qform_mat': 'qform_mat',
            'dwi_reg': 'registered',
            'dwi_reg_mat': 'matrix'})}

    dwi_preproc_pipeline = CombinedStudy.translate(
        'dwi', DiffusionStudy.preprocessing_pipeline)

    dwi_bet_pipeline = CombinedStudy.translate(
        'dwi', DiffusionStudy.brain_mask_pipeline)

    ref_bet_pipeline = CombinedStudy.translate(
        'reference', MRIStudy.brain_mask_pipeline)

    ref_basic_preproc_pipeline = CombinedStudy.translate(
        'reference', MRIStudy.basic_preproc_pipeline,
        override_default_options={'resolution': [1]})

    dwi_qform_transform_pipeline = CombinedStudy.translate(
        'coreg', CoregisteredStudy.qform_transform_pipeline)

    dwi_rigid_registration_pipeline = CombinedStudy.translate(
        'coreg', CoregisteredStudy.linear_registration_pipeline)

    epi_brain_mask_pipeline = CombinedStudy.translate(
        'epi', EPIStudy.brain_mask_pipeline)
    
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