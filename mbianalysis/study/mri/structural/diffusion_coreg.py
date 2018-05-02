from ..base import MRIStudy
from nianalysis.dataset import DatasetSpec, FieldSpec
from mbianalysis.data_format import (
    nifti_gz_format, text_matrix_format, directory_format, dicom_format,
    eddy_par_format, text_format)
from nipype.interfaces.fsl import (ExtractROI, TOPUP, ApplyTOPUP)
from mbianalysis.interfaces.custom.motion_correction import (
    PrepareDWI, CheckDwiNames, GenTopupConfigFiles)
from mbianalysis.citation import fsl_cite
from nianalysis.study.base import StudyMetaClass
from ..coregistered import CoregisteredStudy
from nianalysis.study.multi import (
    MultiStudy, SubStudySpec, MultiStudyMetaClass)
from mbianalysis.interfaces.custom.motion_correction import (
    MotionMatCalculation, AffineMatrixGeneration)
from nipype.interfaces.utility import Merge as merge_lists
from mbianalysis.interfaces.mrtrix.preproc import DWIPreproc
from nipype.interfaces.fsl.utils import Merge as fsl_merge
from mbianalysis.requirement import fsl509_req, mrtrix3_req, fsl510_req
from nianalysis.interfaces.mrtrix import MRConvert
from nianalysis.option import OptionSpec
from nianalysis.exception import NiAnalysisError, NiAnalysisUsageError


class DWIStudy(MRIStudy):

    DWI_PREPROC_NAME = 'dwi_preproc'
    __metaclass__ = StudyMetaClass

    add_option_specs = [
        OptionSpec('bet_robust', True),
        OptionSpec('bet_f_threshold', 0.5),
        OptionSpec('bet_reduce_bias', False),
        OptionSpec('bet_g_threshold', 0.0),
        OptionSpec('bet_method', 'fsl_bet',
                   choices=('fsl_bet', 'optibet')),
        OptionSpec('dwi_preproc_method', 'eddy',
                   choices=('eddy', 'topup'))]

    def dwi1_dcm2nii_conversion_pipeline(self, **kwargs):
        return super(DWIStudy, self).dcm2nii_conversion_pipeline_factory(
            'dwi_main_dcm2nii', 'dwi_1', **kwargs)

    def dwi2_dcm2nii_conversion_pipeline(self, **kwargs):
        return super(DWIStudy, self).dcm2nii_conversion_pipeline_factory(
            'dwi_main_dcm2nii', 'dwi_2', **kwargs)

    def dwi1_header_info_extraction_pipeline(self, **kwargs):
        return (super(DWIStudy, self).
                header_info_extraction_pipeline_factory('dwi_1', **kwargs))

    def dwi2_header_info_extraction_pipeline(self, **kwargs):
        return (super(DWIStudy, self).
                header_info_extraction_pipeline_factory('dwi_2', **kwargs))

    def dwi_preproc_pipeline_factory(self, method='eddy', main=True, **kwargs):
        dwi_preproc_method = self.option('dwi_preproc_method',
                                         self.DWI_PREPROC_NAME)
        if dwi_preproc_method == 'eddy':
            pipeline = self._eddy_dwipreproc_pipeline(**kwargs)
        elif dwi_preproc_method == 'topup':
            pipeline = self._optiBET_brain_mask_pipeline(**kwargs)
        else:
            raise NiAnalysisError("Unrecognised dwi preprocessing method '{}'"
                                  .format(dwi_preproc_method))
        return pipeline

    def _eddy_dwipreproc_pipeline(self, **kwargs):

        pipeline = self.create_pipeline(
            name=self.DWI_PREPROC_NAME,
            inputs=[DatasetSpec('dwi_1', dicom_format),
                    DatasetSpec('dwi_2', dicom_format),
                    FieldSpec('ped', dtype=str),
                    FieldSpec('pe_angle', dtype=str)],
            outputs=[DatasetSpec('preproc', nifti_gz_format),
                     DatasetSpec('eddy_par', eddy_par_format)],
            description=("Diffusion pre-processing pipeline"),
            version=1,
            citations=[],
            **kwargs)

        converter1 = pipeline.create_node(MRConvert(), name='converter1',
                                          requirements=[mrtrix3_req])
        converter1.inputs.out_ext = '.nii.gz'
        pipeline.connect_input('dwi_main', converter1, 'in_file')
        converter2 = pipeline.create_node(MRConvert(), name='converter2',
                                          requirements=[mrtrix3_req])
        converter2.inputs.out_ext = '.nii.gz'
        pipeline.connect_input('dwi_ref', converter2, 'in_file')
        prep_dwi = pipeline.create_node(PrepareDWI(), name='prepare_dwi')
        pipeline.connect_input('ped', prep_dwi, 'pe_dir')
        pipeline.connect_input('pe_angle', prep_dwi, 'phase_offset')
#         prep_dwi.inputs.pe_dir = 'ROW'
#         prep_dwi.inputs.phase_offset = '-1.5'
        pipeline.connect(converter1, 'out_file', prep_dwi, 'dwi')
        pipeline.connect(converter2, 'out_file', prep_dwi, 'dwi1')

        check_name = pipeline.create_node(CheckDwiNames(),
                                          name='check_names')
        pipeline.connect(prep_dwi, 'main', check_name, 'nifti_dwi')
        pipeline.connect_input('dwi_main', check_name, 'dicom_dwi')
        pipeline.connect_input('dwi_ref', check_name, 'dicom_dwi1')
        roi = pipeline.create_node(ExtractROI(), name='extract_roi',
                                   requirements=[fsl509_req])
        roi.inputs.t_min = 0
        roi.inputs.t_size = 1
        pipeline.connect(prep_dwi, 'main', roi, 'in_file')

        merge_outputs = pipeline.create_node(merge_lists(2),
                                             name='merge_files')
        pipeline.connect(roi, 'roi_file', merge_outputs, 'in1')
        pipeline.connect(prep_dwi, 'secondary', merge_outputs, 'in2')
        merge = pipeline.create_node(fsl_merge(), name='fsl_merge',
                                     requirements=[fsl509_req])
        merge.inputs.dimension = 't'
        pipeline.connect(merge_outputs, 'out', merge, 'in_files')
        dwipreproc = pipeline.create_node(
            DWIPreproc(), name='dwipreproc',
            requirements=[fsl510_req, mrtrix3_req])
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

        return pipeline

    def _topup_preproc_pipeline(self, **kwargs):

        pipeline = self.create_pipeline(
            name=self.DWI_PREPROC_NAME,
            inputs=[DatasetSpec('dwi_1_nifti', nifti_gz_format),
                    DatasetSpec('dwi_2_nifti', nifti_gz_format),
                    FieldSpec('pe_dir', dtype=str),
                    FieldSpec('pe_angle', dtype=str)],
            outputs=[DatasetSpec('preproc', nifti_gz_format)],
            description=("Topup distortion correction pipeline."),
            version=1,
            citations=[],
            **kwargs)

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
        merge1 = pipeline.create_node(fsl_merge(), name='fsl_merge1',
                                      requirements=[fsl509_req])
        merge1.inputs.dimension = 't'
        pipeline.connect(merge_outputs1, 'out', merge1, 'in_files')
        topup1 = pipeline.create_node(TOPUP(), name='topup1',
                                      requirements=[fsl509_req])
        pipeline.connect(merge1, 'merged_file', topup1, 'in_file')
        pipeline.connect(ped1, 'config_file', topup1, 'encoding_file')
        in_apply_tp1 = pipeline.create_node(merge_lists(1),
                                            name='in_apply_tp1')
        pipeline.connect_input(to_be_corrected_name, in_apply_tp1, 'in1')
        apply_topup1 = pipeline.create_node(ApplyTOPUP(), name='applytopup1',
                                            requirements=[fsl509_req])
        apply_topup1.inputs.method = 'jac'
        apply_topup1.inputs.in_index = [1]
        pipeline.connect(in_apply_tp1, 'out', apply_topup1, 'in_files')
        pipeline.connect(
            ped1, 'apply_topup_config', apply_topup1, 'encoding_file')
        pipeline.connect(topup1, 'out_movpar', apply_topup1, 'in_topup_movpar')
        pipeline.connect(
            topup1, 'out_fieldcoef', apply_topup1, 'in_topup_fieldcoef')

        pipeline.connect_output(output_name, apply_topup1, 'out_corrected')
        return pipeline


class DiffusionStudy(MRIStudy):

    __metaclass__ = StudyMetaClass

    add_data_specs = [
        DatasetSpec('dwi_main', dicom_format),
        DatasetSpec('dwi_ref', dicom_format),
        DatasetSpec('topup_in', nifti_gz_format),
        DatasetSpec('topup_ref', nifti_gz_format),
        DatasetSpec('preproc', nifti_gz_format, 'dwipreproc_pipeline'),
        DatasetSpec('eddy_par', eddy_par_format, 'dwipreproc_pipeline'),
        DatasetSpec('dwi_distorted', nifti_gz_format, 'topup_pipeline'),
        FieldSpec('ped', str, 'header_info_extraction_pipeline'),
        FieldSpec('pe_angle', str, 'header_info_extraction_pipeline')]

    add_option_specs = [
        OptionSpec('bet_robust', True),
        OptionSpec('bet_f_threshold', 0.2),
        OptionSpec('bet_reduce_bias', False)]

    def dcm2nii_conversion_pipeline(self, **kwargs):
        return super(DiffusionStudy, self).dcm2nii_conversion_pipeline_factory(
            'dwi_main_dcm2nii', 'dwi_main', **kwargs)

    def header_info_extraction_pipeline(self, **kwargs):
        return (super(DiffusionStudy, self).
                header_info_extraction_pipeline_factory('dwi_main', **kwargs))

    def dwipreproc_pipeline(self, **kwargs):

        pipeline = self.create_pipeline(
            name='dwipreproc_pipeline',
            inputs=[DatasetSpec('dwi_main', dicom_format),
                    DatasetSpec('dwi_ref', dicom_format),
                    FieldSpec('ped', dtype=str),
                    FieldSpec('pe_angle', dtype=str)],
            outputs=[DatasetSpec('preproc', nifti_gz_format),
                     DatasetSpec('eddy_par', eddy_par_format)],
            description=("Diffusion pre-processing pipeline"),
            version=1,
            citations=[],
            **kwargs)

        converter1 = pipeline.create_node(MRConvert(), name='converter1',
                                          requirements=[mrtrix3_req])
        converter1.inputs.out_ext = '.nii.gz'
        pipeline.connect_input('dwi_main', converter1, 'in_file')
        converter2 = pipeline.create_node(MRConvert(), name='converter2',
                                          requirements=[mrtrix3_req])
        converter2.inputs.out_ext = '.nii.gz'
        pipeline.connect_input('dwi_ref', converter2, 'in_file')
        prep_dwi = pipeline.create_node(PrepareDWI(), name='prepare_dwi')
        pipeline.connect_input('ped', prep_dwi, 'pe_dir')
        pipeline.connect_input('pe_angle', prep_dwi, 'phase_offset')
#         prep_dwi.inputs.pe_dir = 'ROW'
#         prep_dwi.inputs.phase_offset = '-1.5'
        pipeline.connect(converter1, 'out_file', prep_dwi, 'dwi')
        pipeline.connect(converter2, 'out_file', prep_dwi, 'dwi1')

        check_name = pipeline.create_node(CheckDwiNames(),
                                          name='check_names')
        pipeline.connect(prep_dwi, 'main', check_name, 'nifti_dwi')
        pipeline.connect_input('dwi_main', check_name, 'dicom_dwi')
        pipeline.connect_input('dwi_ref', check_name, 'dicom_dwi1')
        roi = pipeline.create_node(ExtractROI(), name='extract_roi',
                                   requirements=[fsl509_req])
        roi.inputs.t_min = 0
        roi.inputs.t_size = 1
        pipeline.connect(prep_dwi, 'main', roi, 'in_file')

        merge_outputs = pipeline.create_node(merge_lists(2),
                                             name='merge_files')
        pipeline.connect(roi, 'roi_file', merge_outputs, 'in1')
        pipeline.connect(prep_dwi, 'secondary', merge_outputs, 'in2')
        merge = pipeline.create_node(fsl_merge(), name='fsl_merge',
                                     requirements=[fsl509_req])
        merge.inputs.dimension = 't'
        pipeline.connect(merge_outputs, 'out', merge, 'in_files')
        dwipreproc = pipeline.create_node(
            DWIPreproc(), name='dwipreproc',
            requirements=[fsl510_req, mrtrix3_req])
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

        return pipeline

    def topup_factory(self, name, to_be_corrected_name, ref_input_name,
                      pe_dir, pe_angle, output_name, **kwargs):

        pipeline = self.create_pipeline(
            name=name,
            inputs=[DatasetSpec(to_be_corrected_name, nifti_gz_format),
                    DatasetSpec(ref_input_name, nifti_gz_format),
                    FieldSpec(pe_dir, dtype=str),
                    FieldSpec(pe_angle, dtype=str)],
            outputs=[DatasetSpec(output_name, nifti_gz_format)],
            description=("Topup distortion correction pipeline."),
            version=1,
            citations=[],
            **kwargs)

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
        merge1 = pipeline.create_node(fsl_merge(), name='fsl_merge1',
                                      requirements=[fsl509_req])
        merge1.inputs.dimension = 't'
        pipeline.connect(merge_outputs1, 'out', merge1, 'in_files')
        topup1 = pipeline.create_node(TOPUP(), name='topup1',
                                      requirements=[fsl509_req])
        pipeline.connect(merge1, 'merged_file', topup1, 'in_file')
        pipeline.connect(ped1, 'config_file', topup1, 'encoding_file')
        in_apply_tp1 = pipeline.create_node(merge_lists(1),
                                            name='in_apply_tp1')
        pipeline.connect_input(to_be_corrected_name, in_apply_tp1, 'in1')
        apply_topup1 = pipeline.create_node(ApplyTOPUP(), name='applytopup1',
                                            requirements=[fsl509_req])
        apply_topup1.inputs.method = 'jac'
        apply_topup1.inputs.in_index = [1]
        pipeline.connect(in_apply_tp1, 'out', apply_topup1, 'in_files')
        pipeline.connect(
            ped1, 'apply_topup_config', apply_topup1, 'encoding_file')
        pipeline.connect(topup1, 'out_movpar', apply_topup1, 'in_topup_movpar')
        pipeline.connect(
            topup1, 'out_fieldcoef', apply_topup1, 'in_topup_fieldcoef')

        pipeline.connect_output(output_name, apply_topup1, 'out_corrected')
        return pipeline

    def topup_pipeline(self, **kwargs):
        return self.topup_factory(
            'dwi_topup', 'topup_in', 'topup_ref', 'dwi_distorted', 'ped',
            'pe_angle', **kwargs)


class DiffusionReferenceStudy(DiffusionStudy):

    __metaclass__ = StudyMetaClass

    add_data_specs = [
        DatasetSpec('to_be_corrected', dicom_format),
        DatasetSpec('topup_ref', dicom_format),
        DatasetSpec('to_be_corrected_nifti', nifti_gz_format,
                    'main_dcm2nii_conversion_pipeline'),
        DatasetSpec('topup_ref_nifti', nifti_gz_format,
                    'ref_dcm2nii_conversion_pipeline'),
        DatasetSpec('preproc', nifti_gz_format, 'topup_pipeline'),
        FieldSpec('ped', str, 'header_info_extraction_pipeline'),
        FieldSpec('pe_angle', str, 'header_info_extraction_pipeline')]

    def header_info_extraction_pipeline(self, **kwargs):
        return (super(DiffusionReferenceStudy, self).
                header_info_extraction_pipeline_factory(
                    'to_be_corrected', multivol=False, **kwargs))

    def main_dcm2nii_conversion_pipeline(self, **kwargs):
        return (super(DiffusionReferenceStudy, self).
                dcm2nii_conversion_pipeline_factory(
                    'dwi2ref_main_dcm2nii', 'to_be_corrected', **kwargs))

    def ref_dcm2nii_conversion_pipeline(self, **kwargs):
        return (super(DiffusionReferenceStudy, self).
                dcm2nii_conversion_pipeline_factory(
                    'dwi2ref_ref_dcm2nii', 'topup_ref', **kwargs))

    def topup_pipeline(self, **kwargs):
        return super(DiffusionReferenceStudy, self).topup_factory(
            'dwi_ref_topup', 'to_be_corrected_nifti', 'topup_ref_nifti',
            'ped', 'pe_angle', 'preproc', **kwargs)


class DiffusionOppositeStudy(DiffusionReferenceStudy):

    __metaclass__ = StudyMetaClass

    add_data_specs = [
        DatasetSpec('to_be_corrected', dicom_format),
        DatasetSpec('topup_ref', nifti_gz_format),
        DatasetSpec('to_be_corrected_nifti', nifti_gz_format,
                    'main_dcm2nii_conversion_pipeline'),
        DatasetSpec('topup_ref_nifti', nifti_gz_format,
                    'ref_dcm2nii_conversion_pipeline'),
        DatasetSpec('preproc', nifti_gz_format, 'topup_pipeline'),
        FieldSpec('ped', str, 'header_info_extraction_pipeline'),
        FieldSpec('pe_angle', str, 'header_info_extraction_pipeline')]

    def header_info_extraction_pipeline(self, **kwargs):
        return (super(DiffusionReferenceStudy, self).
                header_info_extraction_pipeline_factory(
                    'to_be_corrected', multivol=False, **kwargs))

    def main_dcm2nii_conversion_pipeline(self, **kwargs):
        return (super(DiffusionReferenceStudy, self).
                dcm2nii_conversion_pipeline_factory(
                    'dwi_opposite_main_dcm2nii', 'to_be_corrected', **kwargs))

    def ref_dcm2nii_conversion_pipeline(self, **kwargs):
        return (super(DiffusionReferenceStudy, self).
                dcm2nii_conversion_pipeline_factory(
                    'dwi_opposite_ref_dcm2nii', 'topup_ref', **kwargs))

    def topup_pipeline(self, **kwargs):
        return super(DiffusionOppositeStudy, self).topup_factory(
            'dwi_ref_topup', 'to_be_corrected_nifti', 'topup_ref_nifti',
            'ped', 'pe_angle', 'preproc', **kwargs)


class DiffusionReferenceOppositeStudy(DiffusionReferenceStudy):

    __metaclass__ = StudyMetaClass

    add_data_specs = [
        DatasetSpec('to_be_corrected', dicom_format),
        DatasetSpec('topup_ref', nifti_gz_format),
        DatasetSpec('to_be_corrected_nifti', nifti_gz_format,
                    'main_dcm2nii_conversion_pipeline'),
        DatasetSpec('topup_ref_nifti', nifti_gz_format,
                    'ref_dcm2nii_conversion_pipeline'),
        DatasetSpec('preproc', nifti_gz_format, 'topup_pipeline'),
        FieldSpec('ped', str, 'header_info_extraction_pipeline'),
        FieldSpec('pe_angle', str, 'header_info_extraction_pipeline')]

    def header_info_extraction_pipeline(self, **kwargs):
        return (super(DiffusionReferenceStudy, self).
                header_info_extraction_pipeline_factory(
                    'to_be_corrected', multivol=False, **kwargs))

    def main_dcm2nii_conversion_pipeline(self, **kwargs):
        return (super(DiffusionReferenceStudy, self).
                dcm2nii_conversion_pipeline_factory(
                    'dwi2ref_opp_main_dcm2nii', 'to_be_corrected',
                    **kwargs))

    def ref_dcm2nii_conversion_pipeline(self, **kwargs):
        return (super(DiffusionReferenceStudy, self).
                dcm2nii_conversion_pipeline_factory(
                    'dwi2ref_opp_ref_dcm2nii', 'topup_ref', **kwargs))

    def topup_pipeline(self, **kwargs):
        return super(DiffusionReferenceOppositeStudy, self).topup_factory(
            'dwi_ref_topup', 'to_be_corrected_nifti', 'topup_ref_nifti',
            'ped', 'pe_angle', 'preproc', **kwargs)


class CoregisteredDiffusionStudy(MultiStudy):

    __metaclass__ = MultiStudyMetaClass

    add_sub_study_specs = [
        SubStudySpec('dwi_main', DiffusionStudy, {
            'dwi_main': 'dwi_main',
            'dwi_main_ref': 'dwi_ref',
            'dwi_main_brain_mask': 'brain_mask',
            'dwi_main_brain': 'masked',
            'dwi_main_preproc': 'preproc',
            'dwi_main_eddy_par': 'eddy_par',
            'ped': 'ped',
            'pe_angle': 'pe_angle',
            'tr': 'tr',
            'real_duration': 'real_duration',
            'tot_duration': 'tot_duration',
            'start_time': 'start_time',
            'dcm_info': 'dcm_info'}),
        SubStudySpec('reference', MRIStudy, {
            'reference': 'primary_nifti'}),
        SubStudySpec('coreg', CoregisteredStudy, {
            'dwi_main_brain': 'to_register',
            'reference_masked': 'reference',
            'dwi_main_qformed': 'qformed',
            'dwi_main_qform_mat': 'qform_mat',
            'dwi_main_reg': 'registered',
            'dwi_main_reg_mat': 'matrix'})]

    add_data_specs = [
        DatasetSpec('dwi_main', dicom_format),
        DatasetSpec('dwi_main_ref', dicom_format),
        DatasetSpec('reference', nifti_gz_format),
        DatasetSpec('dwi_main_brain', nifti_gz_format,
                    'dwi_main_bet_pipeline'),
        DatasetSpec('dwi_main_brain_mask', nifti_gz_format,
                    'dwi_main_bet_pipeline'),
        DatasetSpec('dwi_main_preproc', nifti_gz_format,
                    'dwi_main_dwipreproc_pipeline'),
        DatasetSpec('dwi_main_reg', nifti_gz_format,
                    'dwi_main_rigid_registration_pipeline'),
        DatasetSpec('dwi_main_qformed', nifti_gz_format,
                    'dwi_main_qform_transform_pipeline'),
        DatasetSpec('dwi_main_reg_mat', text_matrix_format,
                    'dwi_main_rigid_registration_pipeline'),
        DatasetSpec('dwi_main_qform_mat', text_matrix_format,
                    'dwi_main_qform_transform_pipeline'),
        DatasetSpec('dwi_main_eddy_par', eddy_par_format,
                    'dwi_main_dwipreproc_pipeline'),
        DatasetSpec('motion_mats', directory_format,
                    'motion_mat_pipeline'),
        DatasetSpec('affine_mats', directory_format,
                    'affine_mats_pipeline'),
        DatasetSpec('dcm_info', text_format,
                    'dwi_main_dcm_info_pipeline'),
        FieldSpec('ped', str, 'dwi_main_dcm_info_pipeline'),
        FieldSpec('pe_angle', str, 'dwi_main_dcm_info_pipeline'),
        FieldSpec('tr', float, 'dwi_main_dcm_info_pipeline'),
        FieldSpec('start_time', str, 'dwi_main_dcm_info_pipeline'),
        FieldSpec('real_duration', str, 'dwi_main_dcm_info_pipeline'),
        FieldSpec('tot_duration', str, 'dwi_main_dcm_info_pipeline')]

    add_option_specs = [OptionSpec('reference_resolution', [1])]

    dwi_main_dwipreproc_pipeline = MultiStudy.translate(
        'dwi_main', 'dwipreproc_pipeline')

    dwi_main_bet_pipeline = MultiStudy.translate(
        'dwi_main', 'brain_mask_pipeline')

    dwi_main_dcm_info_pipeline = MultiStudy.translate(
        'dwi_main', 'header_info_extraction_pipeline')

#     ref_bet_pipeline = MultiStudy.translate(
#         'reference', 'brain_mask_pipeline')
#
#     ref_basic_preproc_pipeline = MultiStudy.translate(
#         'reference', 'basic_preproc_pipeline')

    dwi_main_qform_transform_pipeline = MultiStudy.translate(
        'coreg', 'qform_transform_pipeline')

    dwi_main_rigid_registration_pipeline = MultiStudy.translate(
        'coreg', 'linear_registration_pipeline')

    def affine_mats_pipeline(self, **kwargs):

        pipeline = self.create_pipeline(
            name='affine_mat_generation',
            inputs=[DatasetSpec('dwi_main_preproc', nifti_gz_format),
                    DatasetSpec('dwi_main_eddy_par', eddy_par_format)],
            outputs=[
                DatasetSpec('affine_mats', directory_format)],
            description=("Generation of the affine matrices for the main dwi "
                         "sequence starting from eddy motion parameters"),
            version=1,
            citations=[fsl_cite],
            **kwargs)

        aff_mat = pipeline.create_node(AffineMatrixGeneration(),
                                       name='gen_aff_mats')
        pipeline.connect_input('dwi_main_preproc', aff_mat, 'reference_image')
        pipeline.connect_input(
            'dwi_main_eddy_par', aff_mat, 'motion_parameters')
        pipeline.connect_output(
            'affine_mats', aff_mat, 'affine_matrices')
        return pipeline

    def motion_mat_pipeline(self, **kwargs):

        pipeline = self.create_pipeline(
            name='motion_mat_calculation',
            inputs=[DatasetSpec('dwi_main_reg_mat', text_matrix_format),
                    DatasetSpec('dwi_main_qform_mat', text_matrix_format),
                    DatasetSpec('affine_mats', directory_format)],
            outputs=[
                DatasetSpec('motion_mats', directory_format)],
            description=("motion matrices calculation"),
            version=1,
            citations=[fsl_cite],
            **kwargs)

        mm = pipeline.create_node(
            MotionMatCalculation(), name='dwi_main_motion_mats')
        pipeline.connect_input('dwi_main_reg_mat', mm, 'reg_mat')
        pipeline.connect_input('dwi_main_qform_mat', mm, 'qform_mat')
        pipeline.connect_input('affine_mats', mm, 'align_mats')
        pipeline.connect_output('motion_mats', mm, 'motion_mats')
        return pipeline


class CoregisteredDiffusionReferenceStudy(MultiStudy):

    __metaclass__ = MultiStudyMetaClass

    add_sub_study_specs = [
        SubStudySpec('dwi2ref', DiffusionReferenceStudy, {
            'dwi2ref_to_correct': 'to_be_corrected',
            'dwi2ref_ref': 'topup_ref',
            'dwi2ref_to_correct_nii': 'to_be_corrected_nifti',
            'dwi2ref_ref_nii': 'topup_ref_nifti',
            'dwi2ref_brain_mask': 'brain_mask',
            'dwi2ref_brain': 'masked',
            'dwi2ref_preproc': 'preproc',
            'ped': 'ped',
            'pe_angle': 'pe_angle',
            'tr': 'tr',
            'real_duration': 'real_duration',
            'tot_duration': 'tot_duration',
            'start_time': 'start_time',
            'dcm_info': 'dcm_info'}),
        SubStudySpec('reference', MRIStudy, {
            'reference': 'primary_nifti'}),
        SubStudySpec('coreg', CoregisteredStudy, {
            'dwi2ref_brain': 'to_register',
            'reference_masked': 'reference',
            'dwi2ref_qformed': 'qformed',
            'dwi2ref_qform_mat': 'qform_mat',
            'dwi2ref_reg': 'registered',
            'dwi2ref_reg_mat': 'matrix'})]

    add_data_specs = [
        DatasetSpec('dwi2ref_to_correct', dicom_format),
        DatasetSpec('dwi2ref_ref', dicom_format),
        DatasetSpec('reference', nifti_gz_format),
        DatasetSpec('dwi2ref_brain', nifti_gz_format, 'dwi2ref_bet_pipeline'),
        DatasetSpec('dwi2ref_brain_mask', nifti_gz_format,
                    'dwi2ref_bet_pipeline'),
        DatasetSpec('dwi2ref_preproc', nifti_gz_format,
                    'dwi2ref_topup_pipeline'),
        DatasetSpec('dwi2ref_to_correct_nii', nifti_gz_format,
                    'dwi2ref_main_dcm2nii_pipeline'),
        DatasetSpec('dwi2ref_ref_nii', nifti_gz_format,
                    'dwi2ref_ref_dcm2nii_pipeline'),
        DatasetSpec('dwi2ref_reg', nifti_gz_format,
                    'dwi2ref_rigid_registration_pipeline'),
        DatasetSpec('dwi2ref_qformed', nifti_gz_format,
                    'dwi2ref_qform_transform_pipeline'),
        DatasetSpec('dwi2ref_reg_mat', text_matrix_format,
                    'dwi2ref_rigid_registration_pipeline'),
        DatasetSpec('dwi2ref_qform_mat', text_matrix_format,
                    'dwi2ref_qform_transform_pipeline'),
        DatasetSpec('motion_mats', directory_format,
                    'motion_mat_pipeline'),
        DatasetSpec('dcm_info', text_format,
                    'dwi2ref_dcm_info_pipeline'),
        FieldSpec('ped', str, 'dwi2ref_dcm_info_pipeline'),
        FieldSpec('pe_angle', str, 'dwi2ref_dcm_info_pipeline'),
        FieldSpec('tr', float, 'dwi2ref_dcm_info_pipeline'),
        FieldSpec('start_time', str, 'dwi2ref_dcm_info_pipeline'),
        FieldSpec('real_duration', str, 'dwi2ref_dcm_info_pipeline'),
        FieldSpec('tot_duration', str, 'dwi2ref_dcm_info_pipeline')]

    add_option_specs = [OptionSpec('reference_resolution', [1])]

    dwi2ref_topup_pipeline = MultiStudy.translate(
        'dwi2ref', 'topup_pipeline')

    dwi2ref_main_dcm2nii_pipeline = MultiStudy.translate(
        'dwi2ref', 'main_dcm2nii_conversion_pipeline')

    dwi2ref_ref_dcm2nii_pipeline = MultiStudy.translate(
        'dwi2ref', 'ref_dcm2nii_conversion_pipeline')

    dwi2ref_dcm_info_pipeline = MultiStudy.translate(
        'dwi2ref', 'header_info_extraction_pipeline')

    dwi2ref_bet_pipeline = MultiStudy.translate(
        'dwi2ref', 'brain_mask_pipeline')

    ref_bet_pipeline = MultiStudy.translate(
        'reference', 'brain_mask_pipeline')

    ref_basic_preproc_pipeline = MultiStudy.translate(
        'reference', 'basic_preproc_pipeline')

    dwi2ref_qform_transform_pipeline = MultiStudy.translate(
        'coreg', 'qform_transform_pipeline')

    dwi2ref_rigid_registration_pipeline = MultiStudy.translate(
        'coreg', 'linear_registration_pipeline')

    def motion_mat_pipeline(self, **kwargs):

        pipeline = self.create_pipeline(
            name='motion_mat_calculation',
            inputs=[DatasetSpec('dwi2ref_reg_mat', text_matrix_format),
                    DatasetSpec('dwi2ref_qform_mat', text_matrix_format)],
            outputs=[
                DatasetSpec('motion_mats', directory_format)],
            description=("DWI to reference Motion matrices calculation"),
            version=1,
            citations=[fsl_cite],
            **kwargs)

        mm = pipeline.create_node(
            MotionMatCalculation(), name='dwi2ref_motion_mats')
        pipeline.connect_input('dwi2ref_reg_mat', mm, 'reg_mat')
        pipeline.connect_input('dwi2ref_qform_mat', mm, 'qform_mat')
        pipeline.connect_output('motion_mats', mm, 'motion_mats')
        return pipeline


class CoregisteredDiffusionOppositeStudy(MultiStudy):

    __metaclass__ = MultiStudyMetaClass

    add_sub_study_specs = [
        SubStudySpec('dwi_opposite', DiffusionOppositeStudy, {
            'dwi_opposite_to_correct': 'to_be_corrected',
            'dwi_opposite_ref': 'topup_ref',
            'dwi_opposite_to_correct_nii': 'to_be_corrected_nifti',
            'dwi_opposite_ref_nii': 'topup_ref_nifti',
            'dwi_opposite_brain_mask': 'brain_mask',
            'dwi_opposite_brain': 'masked',
            'dwi_opposite_preproc': 'preproc',
            'ped': 'ped',
            'pe_angle': 'pe_angle',
            'tr': 'tr',
            'real_duration': 'real_duration',
            'tot_duration': 'tot_duration',
            'start_time': 'start_time',
            'dcm_info': 'dcm_info'}),
        SubStudySpec('reference', MRIStudy, {
            'reference': 'primary_nifti'}),
        SubStudySpec('coreg', CoregisteredStudy, {
            'dwi_opposite_brain': 'to_register',
            'reference_masked': 'reference',
            'dwi_opposite_qformed': 'qformed',
            'dwi_opposite_qform_mat': 'qform_mat',
            'dwi_opposite_reg': 'registered',
            'dwi_opposite_reg_mat': 'matrix'})]

    add_data_specs = [
        DatasetSpec('dwi_opposite_to_correct', dicom_format),
        DatasetSpec('dwi_opposite_ref', dicom_format),
        DatasetSpec('reference', nifti_gz_format),
        DatasetSpec('dwi_opposite_brain', nifti_gz_format,
                    'dwi_opposite_bet_pipeline'),
        DatasetSpec('dwi_opposite_to_correct_nii', nifti_gz_format,
                    'dwi_opposite_main_dcm2nii_pipeline'),
        DatasetSpec('dwi_opposite_ref_nii', nifti_gz_format,
                    'dwi_opposite_ref_dcm2nii_pipeline'),
        DatasetSpec('dwi_opposite_brain_mask', nifti_gz_format,
                    'dwi_opposite_bet_pipeline'),
        DatasetSpec('dwi_opposite_preproc', nifti_gz_format,
                    'dwi_opposite_topup_pipeline'),
        DatasetSpec('dwi_opposite_reg', nifti_gz_format,
                    'dwi_opposite_rigid_registration_pipeline'),
        DatasetSpec('dwi_opposite_qformed', nifti_gz_format,
                    'dwi_opposite_qform_transform_pipeline'),
        DatasetSpec('dwi_opposite_reg_mat', text_matrix_format,
                    'dwi_opposite_rigid_registration_pipeline'),
        DatasetSpec('dwi_opposite_qform_mat', text_matrix_format,
                    'dwi_opposite_qform_transform_pipeline'),
        DatasetSpec('motion_mats', directory_format,
                    'motion_mat_pipeline'),
        DatasetSpec('dcm_info', text_format,
                    'dwi_opposite_dcm_info_pipeline'),
        FieldSpec('ped', str, 'dwi_opposite_dcm_info_pipeline'),
        FieldSpec('pe_angle', str,
                  'dwi_opposite_dcm_info_pipeline'),
        FieldSpec('tr', float, 'dwi_opposite_dcm_info_pipeline'),
        FieldSpec('start_time', str,
                  'dwi_opposite_dcm_info_pipeline'),
        FieldSpec('real_duration', str,
                  'dwi_opposite_dcm_info_pipeline'),
        FieldSpec('tot_duration', str,
                  'dwi_opposite_dcm_info_pipeline')]

    add_option_specs = [OptionSpec('reference_resolution', [1])]

    dwi_opposite_topup_pipeline = MultiStudy.translate(
        'dwi_opposite', 'topup_pipeline')

    dwi_opposite_main_dcm2nii_pipeline = MultiStudy.translate(
        'dwi_opposite', 'main_dcm2nii_conversion_pipeline')

    dwi_opposite_ref_dcm2nii_pipeline = MultiStudy.translate(
        'dwi_opposite', 'ref_dcm2nii_conversion_pipeline')

    dwi_opposite_dcm_info_pipeline = MultiStudy.translate(
        'dwi_opposite', 'header_info_extraction_pipeline')

    dwi_opposite_bet_pipeline = MultiStudy.translate(
        'dwi_opposite', 'brain_mask_pipeline')

    ref_bet_pipeline = MultiStudy.translate(
        'reference', 'brain_mask_pipeline')

    ref_basic_preproc_pipeline = MultiStudy.translate(
        'reference', 'basic_preproc_pipeline')

    dwi_opposite_qform_transform_pipeline = MultiStudy.translate(
        'coreg', 'qform_transform_pipeline')

    dwi_opposite_rigid_registration_pipeline = MultiStudy.translate(
        'coreg', 'linear_registration_pipeline')

    def motion_mat_pipeline(self, **kwargs):

        pipeline = self.create_pipeline(
            name='motion_mat_calculation',
            inputs=[DatasetSpec('dwi_opposite_reg_mat', text_matrix_format),
                    DatasetSpec('dwi_opposite_qform_mat', text_matrix_format)],
            outputs=[
                DatasetSpec('motion_mats', directory_format)],
            description=("DWI opposite Motion matrices calculation"),
            version=1,
            citations=[fsl_cite],
            **kwargs)

        mm = pipeline.create_node(
            MotionMatCalculation(), name='dwi_opposite_motion_mats')
        pipeline.connect_input('dwi_opposite_reg_mat', mm, 'reg_mat')
        pipeline.connect_input('dwi_opposite_qform_mat', mm, 'qform_mat')
        pipeline.connect_output('motion_mats', mm, 'motion_mats')
        return pipeline


class CoregisteredDiffusionReferenceOppositeStudy(MultiStudy):

    __metaclass__ = MultiStudyMetaClass

    add_sub_study_specs = [
        SubStudySpec('opposite_dwi2ref', DiffusionReferenceOppositeStudy, {
            'opposite_dwi2ref_to_correct': 'to_be_corrected',
            'opposite_dwi2ref_to_correct_nii': 'to_be_corrected_nifti',
            'opposite_dwi2ref_ref': 'topup_ref',
            'opposite_dwi2ref_ref_nii': 'topup_ref_nifti',
            'opposite_dwi2ref_brain_mask': 'brain_mask',
            'opposite_dwi2ref_brain': 'masked',
            'opposite_dwi2ref_preproc': 'preproc',
            'ped': 'ped',
            'pe_angle': 'pe_angle',
            'tr': 'tr',
            'real_duration': 'real_duration',
            'tot_duration': 'tot_duration',
            'start_time': 'start_time',
            'dcm_info': 'dcm_info'}),
        SubStudySpec('reference', MRIStudy, {
            'reference': 'primary_nifti'}),
        SubStudySpec('coreg', CoregisteredStudy, {
            'opposite_dwi2ref_brain': 'to_register',
            'reference_masked': 'reference',
            'opposite_dwi2ref_qformed': 'qformed',
            'opposite_dwi2ref_qform_mat': 'qform_mat',
            'opposite_dwi2ref_reg': 'registered',
            'opposite_dwi2ref_reg_mat': 'matrix'})]

    add_data_specs = [
        DatasetSpec('opposite_dwi2ref_to_correct', dicom_format),
        DatasetSpec('opposite_dwi2ref_ref', dicom_format),
        DatasetSpec('reference', nifti_gz_format),
        DatasetSpec('opposite_dwi2ref_brain', nifti_gz_format,
                    'opposite_dwi2ref_bet_pipeline'),
        DatasetSpec('opposite_dwi2ref_to_correct_nii', nifti_gz_format,
                    'opposite_dwi2ref_main_dcm2nii_pipeline'),
        DatasetSpec('opposite_dwi2ref_ref_nii', nifti_gz_format,
                    'opposite_dwi2ref_ref_dcm2nii_pipeline'),
        DatasetSpec('opposite_dwi2ref_brain_mask', nifti_gz_format,
                    'opposite_dwi2ref_bet_pipeline'),
        DatasetSpec('opposite_dwi2ref_preproc', nifti_gz_format,
                    'opposite_dwi2ref_topup_pipeline'),
        DatasetSpec('opposite_dwi2ref_reg', nifti_gz_format,
                    'opposite_dwi2ref_rigid_registration_pipeline'),
        DatasetSpec('opposite_dwi2ref_qformed', nifti_gz_format,
                    'opposite_dwi2ref_qform_transform_pipeline'),
        DatasetSpec('opposite_dwi2ref_reg_mat', text_matrix_format,
                    'opposite_dwi2ref_rigid_registration_pipeline'),
        DatasetSpec('opposite_dwi2ref_qform_mat', text_matrix_format,
                    'opposite_dwi2ref_qform_transform_pipeline'),
        DatasetSpec('motion_mats', directory_format,
                    'motion_mat_pipeline'),
        DatasetSpec('ref_preproc', nifti_gz_format,
                    'ref_basic_preproc_pipeline'),
        DatasetSpec('ref_brain', nifti_gz_format, 'ref_bet_pipeline'),
        DatasetSpec('ref_brain_mask', nifti_gz_format,
                    'ref_bet_pipeline'),
        DatasetSpec('dcm_info', text_format,
                    'opposite_dwi2ref_dcm_info_pipeline'),
        FieldSpec('ped', str,
                  'opposite_dwi2ref_dcm_info_pipeline'),
        FieldSpec('pe_angle', str,
                  'opposite_dwi2ref_dcm_info_pipeline'),
        FieldSpec('tr', float,
                  'opposite_dwi2ref_dcm_info_pipeline'),
        FieldSpec('start_time', str,
                  'opposite_dwi2ref_dcm_info_pipeline'),
        FieldSpec('real_duration', str,
                  'opposite_dwi2ref_dcm_info_pipeline'),
        FieldSpec('tot_duration', str,
                  'opposite_dwi2ref_dcm_info_pipeline')]

    add_option_specs = [OptionSpec('reference_resolution', [1])]

    opposite_dwi2ref_topup_pipeline = MultiStudy.translate(
        'opposite_dwi2ref', 'topup_pipeline')

    opposite_dwi2ref_main_dcm2nii_pipeline = MultiStudy.translate(
        'opposite_dwi2ref',
        'DiffusionReferenceOppositeStudy_pipeline')

    opposite_dwi2ref_ref_dcm2nii_pipeline = MultiStudy.translate(
        'opposite_dwi2ref',
        'DiffusionReferenceOppositeStudy_pipeline')

    opposite_dwi2ref_dcm_info_pipeline = MultiStudy.translate(
        'opposite_dwi2ref',
        'DiffusionReferenceOppositeStudy_pipeline')

    opposite_dwi2ref_bet_pipeline = MultiStudy.translate(
        'opposite_dwi2ref',
        'DiffusionReferenceOppositeStudy_pipeline')

    ref_bet_pipeline = MultiStudy.translate(
        'reference', 'brain_mask_pipeline')

    ref_basic_preproc_pipeline = MultiStudy.translate(
        'reference', 'basic_preproc_pipeline')

    opposite_dwi2ref_qform_transform_pipeline = MultiStudy.translate(
        'coreg', 'qform_transform_pipeline')

    opposite_dwi2ref_rigid_registration_pipeline = MultiStudy.translate(
        'coreg',
        'CoregisteredStudy_pipeline')

    def motion_mat_pipeline(self, **kwargs):

        pipeline = self.create_pipeline(
            name='opposite_dwi2ref_motion_mat_calculation',
            inputs=[DatasetSpec('opposite_dwi2ref_reg_mat',
                                text_matrix_format),
                    DatasetSpec('opposite_dwi2ref_qform_mat',
                                text_matrix_format)],
            outputs=[
                DatasetSpec('motion_mats', directory_format)],
            description=("DWI to ref opposite Motion matrices calculation"),
            version=1,
            citations=[fsl_cite],
            **kwargs)

        mm = pipeline.create_node(
            MotionMatCalculation(), name='motion_mats')
        pipeline.connect_input('opposite_dwi2ref_reg_mat', mm, 'reg_mat')
        pipeline.connect_input('opposite_dwi2ref_qform_mat', mm, 'qform_mat')
        pipeline.connect_output('motion_mats', mm, 'motion_mats')
        return pipeline
