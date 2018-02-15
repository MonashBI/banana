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
                    FieldSpec('pe_angle', dtype=str)],
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

        pipeline.connect_output('preproc', dwipreproc, 'out_file')
        pipeline.connect_output('eddy_par', dwipreproc, 'eddy_parameters')

        pipeline.assert_connected()
        return pipeline

    def topup_factory(self, name, to_be_corrected_name, ref_input_name,
                      pe_dir, pe_angle, output_name, **options):

        pipeline = self.create_pipeline(
            name=name,
            inputs=[DatasetSpec(to_be_corrected_name, nifti_gz_format),
                    DatasetSpec(ref_input_name, nifti_gz_format),
                    FieldSpec(pe_dir, dtype=str),
                    FieldSpec(pe_angle, dtype=str)],
            outputs=[DatasetSpec(output_name, nifti_gz_format)],
            description=("Topup distortion correction pipeline."),
            default_options={},
            version=1,
            citations=[],
            options=options)

        prep_dwi = pipeline.create_node(PrepareDWI(), name='prepare_dwi')
        prep_dwi.inputs.topup = True
        pipeline.connect_input(pe_dir, prep_dwi, 'pe_dir')
        pipeline.connect_input(pe_angle, prep_dwi, 'phase_offset')
#         prep_dwi.inputs.pe_dir = pe_dir
#         prep_dwi.inputs.phase_offset = pe_angle
        pipeline.connect_input(to_be_corrected_name, prep_dwi, 'dwi')
        pipeline.connect_input(ref_input_name, prep_dwi, 'dwi1')
        ped1 = pipeline.create_node(GenTopupConfigFiles(), name='gen_config1')
        pipeline.connect(prep_dwi, 'pe', ped1, 'ped')
        merge_outputs1 = pipeline.create_node(merge_lists(2),
                                              name='merge_files1')
        pipeline.connect(prep_dwi, 'main', merge_outputs1, 'in1')
        pipeline.connect(prep_dwi, 'secondary', merge_outputs1, 'in2')
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
                                  'dwi_distorted', 'ped', 'pe_angle')

    _data_specs = set_data_specs(
        DatasetSpec('dwi_main', dicom_format),
        DatasetSpec('dwi_ref', dicom_format),
        DatasetSpec('topup_in', nifti_gz_format),
        DatasetSpec('topup_ref', nifti_gz_format),
        DatasetSpec('preproc', nifti_gz_format, dwipreproc_pipeline),
        DatasetSpec('eddy_par', eddy_par_format, dwipreproc_pipeline),
        DatasetSpec('dwi_distorted', nifti_gz_format, topup_pipeline),
        FieldSpec('ped', dtype=str, header_info_extraction_pipeline),
        FieldSpec('pe_angle', dtype=str, header_info_extraction_pipeline),
        inherit_from=MRIStudy.data_specs())


class DiffusionReferenceStudy(DiffusionStudy):

    dcm_in = [DatasetSpec('to_be_corrected', dicom_format)]

    def header_info_extraction_pipeline(self, dcm_in, **kwargs):
        return (super(DiffusionReferenceStudy, self).
                header_info_extraction_pipeline(dcm_in, **kwargs))

    def dcm2nii_conversion_pipeline(self, dcm_in, **options):
        pipeline = self.create_pipeline(
            name='dicom2nifti_coversion',
            inputs=dcm_in,
            outputs=[DatasetSpec('to_be_corrected_nifti', nifti_gz_format)],
            description=("DICOM to NIFTI conversion for topup input"),
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

    inputs = ['dwi_ref_topup', 'to_be_corrected_nifti', 'topup_ref',
              'preproc', 'ped', 'pe_angle']

    def topup_pipeline(self, inputs, **kwargs):
        return super(DiffusionReferenceStudy, self).topup_pipeline(
            inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5],
            **kwargs)

    _data_specs = set_data_specs(
        DatasetSpec('to_be_corrected', dicom_format),
        DatasetSpec('topup_ref', nifti_gz_format),
        DatasetSpec('to_be_corrected_nifti', nifti_gz_format,
                    dcm2nii_conversion_pipeline),
        DatasetSpec('preproc', nifti_gz_format, topup_pipeline),
        FieldSpec('ped', dtype=str, header_info_extraction_pipeline),
        FieldSpec('pe_angle', dtype=str, header_info_extraction_pipeline),
        inherit_from=DiffusionStudy.data_specs())


class DiffusionOppositeStudy(DiffusionReferenceStudy):

    dcm_in = [DatasetSpec('to_be_corrected', dicom_format)]

    def header_info_extraction_pipeline(self, dcm_in, **kwargs):
        return (super(DiffusionOppositeStudy, self).
                header_info_extraction_pipeline(dcm_in, **kwargs))

    def dcm2nii_conversion_pipeline(self, dcm_in, **kwargs):
        return (super(DiffusionOppositeStudy, self).
                dcm2nii_conversion_pipeline(dcm_in, **kwargs))

    inputs = ['dwi_ref_topup', 'to_be_corrected_nifti', 'topup_ref',
              'preproc', 'ped', 'pe_angle']

    def topup_pipeline(self, inputs, **kwargs):
        return super(DiffusionOppositeStudy, self).topup_pipeline(
            inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5],
            **kwargs)

    _data_specs = set_data_specs(
        DatasetSpec('to_be_corrected', dicom_format),
        DatasetSpec('topup_ref', nifti_gz_format),
        DatasetSpec('to_be_corrected_nifti', nifti_gz_format,
                    dcm2nii_conversion_pipeline),
        DatasetSpec('preproc', nifti_gz_format, topup_pipeline),
        FieldSpec('ped', dtype=str, header_info_extraction_pipeline),
        FieldSpec('pe_angle', dtype=str, header_info_extraction_pipeline),
        inherit_from=DiffusionStudy.data_specs())


class DiffusionReferenceOppositeStudy(DiffusionReferenceStudy):

    dcm_in = [DatasetSpec('to_be_corrected', dicom_format)]

    def header_info_extraction_pipeline(self, dcm_in, **kwargs):
        return (super(DiffusionReferenceOppositeStudy, self).
                header_info_extraction_pipeline(dcm_in, **kwargs))

    def dcm2nii_conversion_pipeline(self, dcm_in, **kwargs):
        return (super(DiffusionReferenceOppositeStudy, self).
                dcm2nii_conversion_pipeline(dcm_in, **kwargs))

    inputs = ['dwi_ref_topup', 'to_be_corrected_nifti', 'topup_ref',
              'preproc', 'ped', 'pe_angle']

    def topup_pipeline(self, inputs, **kwargs):
        return super(DiffusionReferenceOppositeStudy, self).topup_pipeline(
            inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5],
            **kwargs)

    _data_specs = set_data_specs(
        DatasetSpec('to_be_corrected', dicom_format),
        DatasetSpec('topup_ref', nifti_gz_format),
        DatasetSpec('to_be_corrected_nifti', nifti_gz_format,
                    dcm2nii_conversion_pipeline),
        DatasetSpec('preproc', nifti_gz_format, topup_pipeline),
        FieldSpec('ped', dtype=str, header_info_extraction_pipeline),
        FieldSpec('pe_angle', dtype=str, header_info_extraction_pipeline),
        inherit_from=DiffusionStudy.data_specs())


class CoregisteredDWIStudy(CombinedStudy):

    sub_study_specs = {
        'dwi_main': (DiffusionStudy, {
            'dwi_main': 'dwi_main',
            'dwi_main_ref': 'dwi_ref',
            'dwi_main_brain_mask': 'brain_mask',
            'dwi_main_brain': 'masked',
            'dwi_main_preproc': 'preproc',
            'dwi_main_eddy_par': 'eddy_par'}),
        'dwi2ref': (DiffusionReferenceStudy, {
            'dwi2ref_to_correct': 'to_be_corrected',
            'dwi2ref_ref': 'topup_ref',
            'dwi2ref_brain_mask': 'brain_mask',
            'dwi2ref_brain': 'masked',
            'dwi2ref_preproc': 'preproc'}),
        'dwi_opposite': (DiffusionOppositeStudy, {
            'dwi_opposite_to_correct': 'to_be_corrected',
            'dwi_opposite_ref': 'topup_ref',
            'dwi_opposite_brain_mask': 'brain_mask',
            'dwi_opposite_brain': 'masked',
            'dwi_opposite_preproc': 'preproc'}),
        'dwi2ref_opposite': (DiffusionReferenceOppositeStudy, {
            'dwi2ref_opposite_to_correct': 'to_be_corrected',
#             'dwi2ref_opposite_to_correct_nii': 'to_be_corrected_nifti',
            'dwi2ref_opposite_ref': 'topup_ref',
            'dwi2ref_opposite_brain_mask': 'brain_mask',
            'dwi2ref_opposite_brain': 'masked',
            'dwi2ref_opposite_preproc': 'preproc'}),
        'reference': (MRIStudy, {
            'reference': 'primary',
            'ref_preproc': 'preproc',
            'ref_brain': 'masked',
            'ref_brain_mask': 'brain_mask'}),
        'coreg_dwi_main': (CoregisteredStudy, {
            'dwi_main_brain': 'to_register',
            'ref_preproc': 'reference',
            'dwi_main_qformed': 'qformed',
            'dwi_main_qform_mat': 'qform_mat',
            'dwi_main_reg': 'registered',
            'dwi_main_reg_mat': 'matrix'}),
        'coreg_dwi2ref': (CoregisteredStudy, {
            'dwi2ref_brain': 'to_register',
            'ref_preproc': 'reference',
            'dwi2ref_qformed': 'qformed',
            'dwi2ref_qform_mat': 'qform_mat',
            'dwi2ref_reg': 'registered',
            'dwi2ref_reg_mat': 'matrix'}),
        'coreg_dwi_opposite': (CoregisteredStudy, {
            'dwi_opposite_brain': 'to_register',
            'ref_preproc': 'reference',
            'dwi_opposite_qformed': 'qformed',
            'dwi_opposite_qform_mat': 'qform_mat',
            'dwi_opposite_reg': 'registered',
            'dwi_opposite_reg_mat': 'matrix'}),
        'coreg_dwi2ref_opposite': (CoregisteredStudy, {
            'dwi2ref_opposite_brain': 'to_register',
            'ref_preproc': 'reference',
            'dwi2ref_opposite_qformed': 'qformed',
            'dwi2ref_opposite_qform_mat': 'qform_mat',
            'dwi2ref_opposite_reg': 'registered',
            'dwi2ref_opposite_reg_mat': 'matrix'})}

    dwi_main_dwipreproc_pipeline = CombinedStudy.translate(
        'dwi_main', DiffusionStudy.dwipreproc_pipeline)

    dwi_main_bet_pipeline = CombinedStudy.translate(
        'dwi_main', DiffusionStudy.brain_mask_pipeline)

    dwi_opposite_topup_pipeline = CombinedStudy.translate(
        'dwi_opposite', DiffusionOppositeStudy.topup_pipeline)

    dwi_opposite_bet_pipeline = CombinedStudy.translate(
        'dwi_opposite', DiffusionOppositeStudy.brain_mask_pipeline)

    dwi2ref_topup_pipeline = CombinedStudy.translate(
        'dwi2ref', DiffusionReferenceStudy.topup_pipeline)

    dwi2ref_bet_pipeline = CombinedStudy.translate(
        'dwi2ref', DiffusionReferenceStudy.brain_mask_pipeline)

    dwi2ref_opposite_topup_pipeline = CombinedStudy.translate(
        'dwi2ref_opposite', DiffusionReferenceOppositeStudy.topup_pipeline)

    dwi2ref_opposite_bet_pipeline = CombinedStudy.translate(
        'dwi2ref_opposite',
        DiffusionReferenceOppositeStudy.brain_mask_pipeline)

#     dwi2ref_opposite_preproc_pipeline = CombinedStudy.translate(
#         'dwi2ref_opposite',
#         DiffusionReferenceOppositeStudy.basic_preproc_pipeline)

    ref_bet_pipeline = CombinedStudy.translate(
        'reference', MRIStudy.brain_mask_pipeline)

    ref_basic_preproc_pipeline = CombinedStudy.translate(
        'reference', MRIStudy.basic_preproc_pipeline,
        override_default_options={'resolution': [1]})

    dwi_main_qform_transform_pipeline = CombinedStudy.translate(
        'coreg_main', CoregisteredStudy.qform_transform_pipeline)

    dwi_main_rigid_registration_pipeline = CombinedStudy.translate(
        'coreg_main', CoregisteredStudy.linear_registration_pipeline)

    dwi_opposite_qform_transform_pipeline = CombinedStudy.translate(
        'coreg_opposite', CoregisteredStudy.qform_transform_pipeline)

    dwi_opposite_rigid_registration_pipeline = CombinedStudy.translate(
        'coreg_opposite', CoregisteredStudy.linear_registration_pipeline)

    dwi2ref_qform_transform_pipeline = CombinedStudy.translate(
        'coreg_dwi2ref', CoregisteredStudy.qform_transform_pipeline)

    dwi2ref_rigid_registration_pipeline = CombinedStudy.translate(
        'coreg_dwi2ref', CoregisteredStudy.linear_registration_pipeline)

    dwi2ref_opposite_qform_transform_pipeline = CombinedStudy.translate(
        'coreg_dwi2ref_opposite', CoregisteredStudy.qform_transform_pipeline)

    dwi2ref_opposite_rigid_registration_pipeline = CombinedStudy.translate(
        'coreg_dwi2ref_opposite',
        CoregisteredStudy.linear_registration_pipeline)

    def dwi_main_motion_mat_pipeline(self, **options):

        pipeline = self.create_pipeline(
            name='dwi2ref_motion_mat_calculation',
            inputs=[DatasetSpec('dwi2ref_reg_mat', text_matrix_format),
                    DatasetSpec('dwi2ref_qform_mat', text_matrix_format)],
            outputs=[
                DatasetSpec('dwi2ref_motion_mats', directory_format)],
            description=("DWI to reference Motion matrices calculation"),
            default_options={},
            version=1,
            citations=[fsl_cite],
            options=options)

        
        pipeline.assert_connected()
        return pipeline

    def dwi_opposite_motion_mat_pipeline(self, **options):

        pipeline = self.create_pipeline(
            name='dwi_opposite_motion_mat_calculation',
            inputs=[DatasetSpec('dwi_opposite_reg_mat', text_matrix_format),
                    DatasetSpec('dwi_opposite_qform_mat', text_matrix_format)],
            outputs=[
                DatasetSpec('dwi_opposite_motion_mats', directory_format)],
            description=("DWI opposite Motion matrices calculation"),
            default_options={},
            version=1,
            citations=[fsl_cite],
            options=options)

        mm = pipeline.create_node(
            MotionMatCalculation(), name='dwi_opposite_motion_mats')
        pipeline.connect_input('dwi_opposite_reg_mat', mm, 'reg_mat')
        pipeline.connect_input('dwi_opposite_qform_mat', mm, 'qform_mat')
        pipeline.connect_output('dwi_opposite_motion_mats', mm, 'motion_mats')
        pipeline.assert_connected()
        return pipeline

    def dwi2ref_motion_mat_pipeline(self, **options):

        pipeline = self.create_pipeline(
            name='dwi2ref_motion_mat_calculation',
            inputs=[DatasetSpec('dwi2ref_reg_mat', text_matrix_format),
                    DatasetSpec('dwi2ref_qform_mat', text_matrix_format)],
            outputs=[
                DatasetSpec('dwi2ref_motion_mats', directory_format)],
            description=("DWI to reference Motion matrices calculation"),
            default_options={},
            version=1,
            citations=[fsl_cite],
            options=options)

        mm = pipeline.create_node(
            MotionMatCalculation(), name='dwi2ref_motion_mats')
        pipeline.connect_input('dwidwi2ref_reg_mat', mm, 'reg_mat')
        pipeline.connect_input('dwidwi2ref_qform_mat', mm, 'qform_mat')
        pipeline.connect_output('dwidwi2ref_motion_mats', mm, 'motion_mats')
        pipeline.assert_connected()
        return pipeline

    def dwi2ref_opposite_motion_mat_pipeline(self, **options):

        pipeline = self.create_pipeline(
            name='dwi2ref_opposite_motion_mat_calculation',
            inputs=[DatasetSpec('dwi2ref_opposite_reg_mat',
                                text_matrix_format),
                    DatasetSpec('dwi2ref_opposite_qform_mat',
                                text_matrix_format)],
            outputs=[
                DatasetSpec('dwi2ref_opposite_motion_mats', directory_format)],
            description=("DWI to ref opposite Motion matrices calculation"),
            default_options={},
            version=1,
            citations=[fsl_cite],
            options=options)

        mm = pipeline.create_node(
            MotionMatCalculation(), name='dwi2ref_opposite_motion_mats')
        pipeline.connect_input('dwi2ref_opposite_reg_mat', mm, 'reg_mat')
        pipeline.connect_input('dwi2ref_opposite_qform_mat', mm, 'qform_mat')
        pipeline.connect_output(
            'dwi2ref_opposite_motion_mats', mm, 'motion_mats')
        pipeline.assert_connected()
        return pipeline

    _data_specs = set_data_specs(
        DatasetSpec('dwi_main', dicom_format),
        DatasetSpec('dwi_main_ref', dicom_format),
        DatasetSpec('dwi2ref_to_correct', dicom_format),
        DatasetSpec('dwi2ref_ref', nifti_gz_format),
        DatasetSpec('dwi_opposite_to_correct', dicom_format),
        DatasetSpec('dwi_opposite_ref', nifti_gz_format),
        DatasetSpec('dwi2ref_opposite_to_correct', dicom_format),
        DatasetSpec('dwi2ref_opposite_ref', nifti_gz_format),
        DatasetSpec('reference', nifti_gz_format),
        DatasetSpec('dwi_main_brain', nifti_gz_format, dwi_main_bet_pipeline),
        DatasetSpec('dwi_main_brain_mask', nifti_gz_format,
                    dwi_main_bet_pipeline),
        DatasetSpec('dwi_main_preproc', nifti_gz_format,
                    dwi_main_dwipreproc_pipeline),
        DatasetSpec('dwi_main_reg', nifti_gz_format,
                    dwi_main_rigid_registration_pipeline),
        DatasetSpec('dwi_main_qformed', nifti_gz_format,
                    dwi_main_qform_transform_pipeline),
        DatasetSpec('dwi_main_reg_mat', text_matrix_format,
                    dwi_main_rigid_registration_pipeline),
        DatasetSpec('dwi_main_qform_mat', text_matrix_format,
                    dwi_main_qform_transform_pipeline),
        DatasetSpec('dwi_main_eddy_par', eddy_par_format,
                    dwi_main_dwipreproc_pipeline),
        DatasetSpec('dwi_main_motion_mats', directory_format,
                    dwi_main_motion_mat_pipeline),
        DatasetSpec('dwi2ref_brain', nifti_gz_format, dwi2ref_bet_pipeline),
        DatasetSpec('dwi2ref_brain_mask', nifti_gz_format,
                    dwi2ref_bet_pipeline),
        DatasetSpec('dwi2ref_preproc', nifti_gz_format,
                    dwi2ref_topup_pipeline),
        DatasetSpec('dwi2ref_reg', nifti_gz_format,
                    dwi2ref_rigid_registration_pipeline),
        DatasetSpec('dwi2ref_qformed', nifti_gz_format,
                    dwi2ref_qform_transform_pipeline),
        DatasetSpec('dwi2ref_reg_mat', text_matrix_format,
                    dwi2ref_rigid_registration_pipeline),
        DatasetSpec('dwi2ref_qform_mat', text_matrix_format,
                    dwi2ref_qform_transform_pipeline),
        DatasetSpec('dwi2ref_motion_mats', directory_format,
                    dwi2ref_motion_mat_pipeline),
        DatasetSpec('dwi_opposite_brain', nifti_gz_format,
                    dwi_opposite_bet_pipeline),
        DatasetSpec('dwi_opposite_brain_mask', nifti_gz_format,
                    dwi_opposite_bet_pipeline),
        DatasetSpec('dwi_opposite_preproc', nifti_gz_format,
                    dwi_opposite_topup_pipeline),
        DatasetSpec('dwi_opposite_reg', nifti_gz_format,
                    dwi_opposite_rigid_registration_pipeline),
        DatasetSpec('dwi_opposite_qformed', nifti_gz_format,
                    dwi_opposite_qform_transform_pipeline),
        DatasetSpec('dwi_opposite_reg_mat', text_matrix_format,
                    dwi_opposite_rigid_registration_pipeline),
        DatasetSpec('dwi_opposite_qform_mat', text_matrix_format,
                    dwi_opposite_qform_transform_pipeline),
        DatasetSpec('dwi_opposite_motion_mats', directory_format,
                    dwi_opposite_motion_mat_pipeline),
        DatasetSpec('dwi2ref_opposite_brain', nifti_gz_format,
                    dwi2ref_opposite_bet_pipeline),
        DatasetSpec('dwi2ref_opposite_brain_mask', nifti_gz_format,
                    dwi2ref_opposite_bet_pipeline),
        DatasetSpec('dwi2ref_opposite_preproc', nifti_gz_format,
                    dwi2ref_opposite_topup_pipeline),
        DatasetSpec('dwi2ref_opposite_reg', nifti_gz_format,
                    dwi2ref_opposite_rigid_registration_pipeline),
        DatasetSpec('dwi2ref_opposite_qformed', nifti_gz_format,
                    dwi2ref_opposite_qform_transform_pipeline),
        DatasetSpec('dwi2ref_opposite_reg_mat', text_matrix_format,
                    dwi2ref_opposite_rigid_registration_pipeline),
        DatasetSpec('dwi2ref_opposite_qform_mat', text_matrix_format,
                    dwi2ref_opposite_qform_transform_pipeline),
        DatasetSpec('dwi2ref_opposite_motion_mats', directory_format,
                    dwi2ref_opposite_motion_mat_pipeline),
        DatasetSpec('ref_preproc', nifti_gz_format,
                    ref_basic_preproc_pipeline),
        DatasetSpec('ref_brain', nifti_gz_format, ref_bet_pipeline),
        DatasetSpec('ref_brain_mask', nifti_gz_format,
                    ref_bet_pipeline))
