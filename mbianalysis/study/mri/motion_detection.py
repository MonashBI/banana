from nianalysis.dataset import DatasetSpec, FieldSpec
from nianalysis.data_formats import (
    nifti_gz_format, text_matrix_format, directory_format, dicom_format,
    par_format, text_format)
from mbianalysis.interfaces.custom.motion_correction import (
    MeanDisplacementCalculation, MotionFraming)
from nianalysis.citations import fsl_cite
from nianalysis.study.base import set_specs
from nianalysis.study.multi import (
    MultiStudy, SubStudySpec, MultiStudyMetaClass)
from .epi import CoregisteredEPIStudy
from .structural.t1 import CoregisteredT1Study
from .structural.t2 import CoregisteredT2Study
from nipype.interfaces.utility import Merge as merge_lists
from .base import MotionReferenceStudy


class MotionDetectionStudy(MultiStudy):

    __metaclass__ = MultiStudyMetaClass

    ref_dcm_info_pipeline = MultiStudy.translate(
        'ref', MotionReferenceStudy.header_info_extraction_pipeline)

    t1_motion_alignment_pipeline = MultiStudy.translate(
        't1_1', CoregisteredT1Study.t1_motion_mat_pipeline)

    t1_dcm2nii_pipeline = MultiStudy.translate(
        't1_1', CoregisteredT1Study.t1_dcm2nii_pipeline)

    t1_dcm_info_pipeline = MultiStudy.translate(
        't1_1', CoregisteredT1Study.t1_dcm_info_pipeline)

    t1_motion_mat_pipeline = MultiStudy.translate(
        't1_1', CoregisteredT1Study.t1_motion_mat_pipeline)

    t1_basic_preproc_pipeline = MultiStudy.translate(
        't1_1', CoregisteredT1Study.t1_basic_preproc_pipeline)

    t1_qform_transform_pipeline = MultiStudy.translate(
        't1_1', CoregisteredT1Study.t1_qform_transform_pipeline)

    t1_bet_pipeline = MultiStudy.translate(
        't1_1', CoregisteredT1Study.t1_bet_pipeline)

    t1_ref_bet_pipeline = MultiStudy.translate(
        't1_1', CoregisteredT1Study.ref_bet_pipeline)

    t1_ref_basic_preproc_pipeline = MultiStudy.translate(
        't1_1', CoregisteredT1Study.ref_basic_preproc_pipeline)

    t1_rigid_registration_pipeline = MultiStudy.translate(
        't1_1', CoregisteredT1Study.t1_rigid_registration_pipeline)

    ute_motion_alignment_pipeline = MultiStudy.translate(
        'ute', CoregisteredT1Study.t1_motion_mat_pipeline)

    ute_dcm2nii_pipeline = MultiStudy.translate(
        'ute', CoregisteredT1Study.t1_dcm2nii_pipeline)

    ute_dcm_info_pipeline = MultiStudy.translate(
        'ute', CoregisteredT1Study.t1_dcm_info_pipeline)

    ute_motion_mat_pipeline = MultiStudy.translate(
        'ute', CoregisteredT1Study.t1_motion_mat_pipeline)

    ute_basic_preproc_pipeline = MultiStudy.translate(
        'ute', CoregisteredT1Study.t1_basic_preproc_pipeline)

    ute_qform_transform_pipeline = MultiStudy.translate(
        'ute', CoregisteredT1Study.t1_qform_transform_pipeline)

    ute_bet_pipeline = MultiStudy.translate(
        'ute', CoregisteredT1Study.t1_bet_pipeline)

    ute_ref_bet_pipeline = MultiStudy.translate(
        'ute', CoregisteredT1Study.ref_bet_pipeline)

    ute_ref_basic_preproc_pipeline = MultiStudy.translate(
        'ute', CoregisteredT1Study.ref_basic_preproc_pipeline)

    ute_rigid_registration_pipeline = MultiStudy.translate(
        'ute', CoregisteredT1Study.t1_rigid_registration_pipeline)

    fm_motion_alignment_pipeline = MultiStudy.translate(
        'fm', CoregisteredT2Study.t2_motion_mat_pipeline)

    fm_dcm2nii_pipeline = MultiStudy.translate(
        'fm', CoregisteredT2Study.t2_dcm2nii_pipeline)

    fm_dcm_info_pipeline = MultiStudy.translate(
        'fm', CoregisteredT2Study.t2_dcm_info_pipeline)

    fm_motion_mat_pipeline = MultiStudy.translate(
        'fm', CoregisteredT2Study.t2_motion_mat_pipeline)

    fm_basic_preproc_pipeline = MultiStudy.translate(
        'fm', CoregisteredT2Study.t2_basic_preproc_pipeline)

    fm_qform_transform_pipeline = MultiStudy.translate(
        'fm', CoregisteredT2Study.t2_qform_transform_pipeline)

    fm_bet_pipeline = MultiStudy.translate(
        'fm', CoregisteredT2Study.t2_bet_pipeline)

    fm_ref_bet_pipeline = MultiStudy.translate(
        'fm', CoregisteredT2Study.ref_bet_pipeline)

    fm_ref_basic_preproc_pipeline = MultiStudy.translate(
        'fm', CoregisteredT2Study.ref_basic_preproc_pipeline)

    fm_rigid_registration_pipeline = MultiStudy.translate(
        'fm', CoregisteredT2Study.t2_rigid_registration_pipeline)

    epi1_motion_alignment_pipeline = MultiStudy.translate(
        'epi1', CoregisteredEPIStudy.epi_motion_alignment_pipeline)

    epi1_dcm2nii_pipeline = MultiStudy.translate(
        'epi1', CoregisteredEPIStudy.epi_dcm2nii_pipeline)

    epi1_epireg_pipeline = MultiStudy.translate(
        'epi1', CoregisteredEPIStudy.epireg_pipeline)

    epi1_dcm_info_pipeline = MultiStudy.translate(
        'epi1', CoregisteredEPIStudy.epi_dcm_info_pipeline)

    epi1_motion_mat_pipeline = MultiStudy.translate(
        'epi1', CoregisteredEPIStudy.epi_motion_mat_pipeline)

    epi1_basic_preproc_pipeline = MultiStudy.translate(
        'epi1', CoregisteredEPIStudy.epi_basic_preproc_pipeline)

    epi1_qform_transform_pipeline = MultiStudy.translate(
        'epi1', CoregisteredEPIStudy.epi_qform_transform_pipeline)

    epi1_bet_pipeline = MultiStudy.translate(
        'epi1', CoregisteredEPIStudy.epi_bet_pipeline)

    epi1_ref_bet_pipeline = MultiStudy.translate(
        'epi1', CoregisteredEPIStudy.ref_bet_pipeline)

    epi1_ref_segmentation_pipeline = MultiStudy.translate(
        'epi1', CoregisteredEPIStudy.ref_segmentation_pipeline)

    epi1_ref_basic_preproc_pipeline = MultiStudy.translate(
        'epi1', CoregisteredEPIStudy.ref_basic_preproc_pipeline)

    epi1_ref_nifti_pipeline = MultiStudy.translate(
        'epi1', CoregisteredEPIStudy.ref_dcm2nii_pipeline)

    epi2_motion_alignment_pipeline = MultiStudy.translate(
        'epi2', CoregisteredEPIStudy.epi_motion_alignment_pipeline)

    epi2_dcm_info_pipeline = MultiStudy.translate(
        'epi2', CoregisteredEPIStudy.epi_dcm_info_pipeline)

    epi2_epireg_pipeline = MultiStudy.translate(
        'epi2', CoregisteredEPIStudy.epireg_pipeline)

    epi2_motion_mat_pipeline = MultiStudy.translate(
        'epi2', CoregisteredEPIStudy.epi_motion_mat_pipeline)

    epi2_basic_preproc_pipeline = MultiStudy.translate(
        'epi2', CoregisteredEPIStudy.epi_basic_preproc_pipeline)

    epi2_qform_transform_pipeline = MultiStudy.translate(
        'epi2', CoregisteredEPIStudy.epi_qform_transform_pipeline)

    epi2_ref_bet_pipeline = MultiStudy.translate(
        'epi2', CoregisteredEPIStudy.ref_bet_pipeline)

    epi2_ref_nifti_pipeline = MultiStudy.translate(
        'epi2', CoregisteredEPIStudy.ref_dcm2nii_pipeline)

    epi2_ref_segmentation_pipeline = MultiStudy.translate(
        'epi2', CoregisteredEPIStudy.ref_segmentation_pipeline)

    epi2_ref_basic_preproc_pipeline = MultiStudy.translate(
        'epi2', CoregisteredEPIStudy.ref_basic_preproc_pipeline)

    epi2_bet_pipeline = MultiStudy.translate(
        'epi2', CoregisteredEPIStudy.epi_bet_pipeline)

    epi2_dcm2nii_pipeline = MultiStudy.translate(
        'epi2', CoregisteredEPIStudy.epi_dcm2nii_pipeline)

    epi3_motion_alignment_pipeline = MultiStudy.translate(
        'epi3', CoregisteredEPIStudy.epi_motion_alignment_pipeline)

    epi3_dcm2nii_pipeline = MultiStudy.translate(
        'epi3', CoregisteredEPIStudy.epi_dcm2nii_pipeline)

    epi3_epireg_pipeline = MultiStudy.translate(
        'epi3', CoregisteredEPIStudy.epireg_pipeline)

    epi3_dcm_info_pipeline = MultiStudy.translate(
        'epi3', CoregisteredEPIStudy.epi_dcm_info_pipeline)

    epi3_motion_mat_pipeline = MultiStudy.translate(
        'epi3', CoregisteredEPIStudy.epi_motion_mat_pipeline)

    epi3_basic_preproc_pipeline = MultiStudy.translate(
        'epi3', CoregisteredEPIStudy.epi_basic_preproc_pipeline)

    epi3_qform_transform_pipeline = MultiStudy.translate(
        'epi3', CoregisteredEPIStudy.epi_qform_transform_pipeline)

    epi3_bet_pipeline = MultiStudy.translate(
        'epi3', CoregisteredEPIStudy.epi_bet_pipeline)

    epi3_ref_bet_pipeline = MultiStudy.translate(
        'epi3', CoregisteredEPIStudy.ref_bet_pipeline)

    epi3_ref_segmentation_pipeline = MultiStudy.translate(
        'epi3', CoregisteredEPIStudy.ref_segmentation_pipeline)

    epi3_ref_basic_preproc_pipeline = MultiStudy.translate(
        'epi3', CoregisteredEPIStudy.ref_basic_preproc_pipeline)

    epi3_ref_nifti_pipeline = MultiStudy.translate(
        'epi3', CoregisteredEPIStudy.ref_dcm2nii_pipeline)

    epi4_motion_alignment_pipeline = MultiStudy.translate(
        'epi4', CoregisteredEPIStudy.epi_motion_alignment_pipeline)

    epi4_dcm2nii_pipeline = MultiStudy.translate(
        'epi4', CoregisteredEPIStudy.epi_dcm2nii_pipeline)

    epi4_epireg_pipeline = MultiStudy.translate(
        'epi4', CoregisteredEPIStudy.epireg_pipeline)

    epi4_dcm_info_pipeline = MultiStudy.translate(
        'epi4', CoregisteredEPIStudy.epi_dcm_info_pipeline)

    epi4_motion_mat_pipeline = MultiStudy.translate(
        'epi4', CoregisteredEPIStudy.epi_motion_mat_pipeline)

    epi4_basic_preproc_pipeline = MultiStudy.translate(
        'epi4', CoregisteredEPIStudy.epi_basic_preproc_pipeline)

    epi4_qform_transform_pipeline = MultiStudy.translate(
        'epi4', CoregisteredEPIStudy.epi_qform_transform_pipeline)

    epi4_bet_pipeline = MultiStudy.translate(
        'epi4', CoregisteredEPIStudy.epi_bet_pipeline)

    epi4_ref_bet_pipeline = MultiStudy.translate(
        'epi4', CoregisteredEPIStudy.ref_bet_pipeline)

    epi4_ref_segmentation_pipeline = MultiStudy.translate(
        'epi4', CoregisteredEPIStudy.ref_segmentation_pipeline)

    epi4_ref_basic_preproc_pipeline = MultiStudy.translate(
        'epi4', CoregisteredEPIStudy.ref_basic_preproc_pipeline)

    epi4_ref_nifti_pipeline = MultiStudy.translate(
        'epi4', CoregisteredEPIStudy.ref_dcm2nii_pipeline)

    epi5_motion_alignment_pipeline = MultiStudy.translate(
        'epi5', CoregisteredEPIStudy.epi_motion_alignment_pipeline)

    epi5_dcm2nii_pipeline = MultiStudy.translate(
        'epi5', CoregisteredEPIStudy.epi_dcm2nii_pipeline)

    epi5_epireg_pipeline = MultiStudy.translate(
        'epi5', CoregisteredEPIStudy.epireg_pipeline)

    epi5_dcm_info_pipeline = MultiStudy.translate(
        'epi5', CoregisteredEPIStudy.epi_dcm_info_pipeline)

    epi5_motion_mat_pipeline = MultiStudy.translate(
        'epi5', CoregisteredEPIStudy.epi_motion_mat_pipeline)

    epi5_basic_preproc_pipeline = MultiStudy.translate(
        'epi5', CoregisteredEPIStudy.epi_basic_preproc_pipeline)

    epi5_qform_transform_pipeline = MultiStudy.translate(
        'epi5', CoregisteredEPIStudy.epi_qform_transform_pipeline)

    epi5_bet_pipeline = MultiStudy.translate(
        'epi5', CoregisteredEPIStudy.epi_bet_pipeline)

    epi5_ref_bet_pipeline = MultiStudy.translate(
        'epi5', CoregisteredEPIStudy.ref_bet_pipeline)

    epi5_ref_segmentation_pipeline = MultiStudy.translate(
        'epi5', CoregisteredEPIStudy.ref_segmentation_pipeline)

    epi5_ref_basic_preproc_pipeline = MultiStudy.translate(
        'epi5', CoregisteredEPIStudy.ref_basic_preproc_pipeline)

    epi5_ref_nifti_pipeline = MultiStudy.translate(
        'epi5', CoregisteredEPIStudy.ref_dcm2nii_pipeline)

    epi6_motion_alignment_pipeline = MultiStudy.translate(
        'epi6', CoregisteredEPIStudy.epi_motion_alignment_pipeline)

    epi6_dcm2nii_pipeline = MultiStudy.translate(
        'epi6', CoregisteredEPIStudy.epi_dcm2nii_pipeline)

    epi6_epireg_pipeline = MultiStudy.translate(
        'epi6', CoregisteredEPIStudy.epireg_pipeline)

    epi6_dcm_info_pipeline = MultiStudy.translate(
        'epi6', CoregisteredEPIStudy.epi_dcm_info_pipeline)

    epi6_motion_mat_pipeline = MultiStudy.translate(
        'epi6', CoregisteredEPIStudy.epi_motion_mat_pipeline)

    epi6_basic_preproc_pipeline = MultiStudy.translate(
        'epi6', CoregisteredEPIStudy.epi_basic_preproc_pipeline)

    epi6_qform_transform_pipeline = MultiStudy.translate(
        'epi6', CoregisteredEPIStudy.epi_qform_transform_pipeline)

    epi6_bet_pipeline = MultiStudy.translate(
        'epi6', CoregisteredEPIStudy.epi_bet_pipeline)

    epi6_ref_bet_pipeline = MultiStudy.translate(
        'epi6', CoregisteredEPIStudy.ref_bet_pipeline)

    epi6_ref_segmentation_pipeline = MultiStudy.translate(
        'epi6', CoregisteredEPIStudy.ref_segmentation_pipeline)

    epi6_ref_basic_preproc_pipeline = MultiStudy.translate(
        'epi6', CoregisteredEPIStudy.ref_basic_preproc_pipeline)

    epi6_ref_nifti_pipeline = MultiStudy.translate(
        'epi6', CoregisteredEPIStudy.ref_dcm2nii_pipeline)

    epi7_motion_alignment_pipeline = MultiStudy.translate(
        'epi7', CoregisteredEPIStudy.epi_motion_alignment_pipeline)

    epi7_dcm2nii_pipeline = MultiStudy.translate(
        'epi7', CoregisteredEPIStudy.epi_dcm2nii_pipeline)

    epi7_epireg_pipeline = MultiStudy.translate(
        'epi7', CoregisteredEPIStudy.epireg_pipeline)

    epi7_dcm_info_pipeline = MultiStudy.translate(
        'epi7', CoregisteredEPIStudy.epi_dcm_info_pipeline)

    epi7_motion_mat_pipeline = MultiStudy.translate(
        'epi7', CoregisteredEPIStudy.epi_motion_mat_pipeline)

    epi7_basic_preproc_pipeline = MultiStudy.translate(
        'epi7', CoregisteredEPIStudy.epi_basic_preproc_pipeline)

    epi7_qform_transform_pipeline = MultiStudy.translate(
        'epi7', CoregisteredEPIStudy.epi_qform_transform_pipeline)

    epi7_bet_pipeline = MultiStudy.translate(
        'epi7', CoregisteredEPIStudy.epi_bet_pipeline)

    epi7_ref_bet_pipeline = MultiStudy.translate(
        'epi7', CoregisteredEPIStudy.ref_bet_pipeline)

    epi7_ref_segmentation_pipeline = MultiStudy.translate(
        'epi7', CoregisteredEPIStudy.ref_segmentation_pipeline)

    epi7_ref_basic_preproc_pipeline = MultiStudy.translate(
        'epi7', CoregisteredEPIStudy.ref_basic_preproc_pipeline)

    epi7_ref_nifti_pipeline = MultiStudy.translate(
        'epi7', CoregisteredEPIStudy.ref_dcm2nii_pipeline)

    epi8_motion_alignment_pipeline = MultiStudy.translate(
        'epi8', CoregisteredEPIStudy.epi_motion_alignment_pipeline)

    epi8_dcm2nii_pipeline = MultiStudy.translate(
        'epi8', CoregisteredEPIStudy.epi_dcm2nii_pipeline)

    epi8_epireg_pipeline = MultiStudy.translate(
        'epi8', CoregisteredEPIStudy.epireg_pipeline)

    epi8_dcm_info_pipeline = MultiStudy.translate(
        'epi8', CoregisteredEPIStudy.epi_dcm_info_pipeline)

    epi8_motion_mat_pipeline = MultiStudy.translate(
        'epi8', CoregisteredEPIStudy.epi_motion_mat_pipeline)

    epi8_basic_preproc_pipeline = MultiStudy.translate(
        'epi8', CoregisteredEPIStudy.epi_basic_preproc_pipeline)

    epi8_qform_transform_pipeline = MultiStudy.translate(
        'epi8', CoregisteredEPIStudy.epi_qform_transform_pipeline)

    epi8_bet_pipeline = MultiStudy.translate(
        'epi8', CoregisteredEPIStudy.epi_bet_pipeline)

    epi8_ref_bet_pipeline = MultiStudy.translate(
        'epi8', CoregisteredEPIStudy.ref_bet_pipeline)

    epi8_ref_segmentation_pipeline = MultiStudy.translate(
        'epi8', CoregisteredEPIStudy.ref_segmentation_pipeline)

    epi8_ref_basic_preproc_pipeline = MultiStudy.translate(
        'epi8', CoregisteredEPIStudy.ref_basic_preproc_pipeline)

    epi8_ref_nifti_pipeline = MultiStudy.translate(
        'epi8', CoregisteredEPIStudy.ref_dcm2nii_pipeline)

    asl_motion_alignment_pipeline = MultiStudy.translate(
        'asl', CoregisteredEPIStudy.epi_motion_alignment_pipeline)

    asl_dcm2nii_pipeline = MultiStudy.translate(
        'asl', CoregisteredEPIStudy.epi_dcm2nii_pipeline)

    asl_epireg_pipeline = MultiStudy.translate(
        'asl', CoregisteredEPIStudy.epireg_pipeline)

    asl_dcm_info_pipeline = MultiStudy.translate(
        'asl', CoregisteredEPIStudy.epi_dcm_info_pipeline)

    asl_motion_mat_pipeline = MultiStudy.translate(
        'asl', CoregisteredEPIStudy.epi_motion_mat_pipeline)

    asl_basic_preproc_pipeline = MultiStudy.translate(
        'asl', CoregisteredEPIStudy.epi_basic_preproc_pipeline)

    asl_qform_transform_pipeline = MultiStudy.translate(
        'asl', CoregisteredEPIStudy.epi_qform_transform_pipeline)

    asl_bet_pipeline = MultiStudy.translate(
        'asl', CoregisteredEPIStudy.epi_bet_pipeline)

    asl_ref_bet_pipeline = MultiStudy.translate(
        'asl', CoregisteredEPIStudy.ref_bet_pipeline)

    asl_ref_segmentation_pipeline = MultiStudy.translate(
        'asl', CoregisteredEPIStudy.ref_segmentation_pipeline)

    asl_ref_basic_preproc_pipeline = MultiStudy.translate(
        'asl', CoregisteredEPIStudy.ref_basic_preproc_pipeline)

    asl_ref_nifti_pipeline = MultiStudy.translate(
        'asl', CoregisteredEPIStudy.ref_dcm2nii_pipeline)

    def mean_displacement_pipeline(self, **options):

        pipeline = self.create_pipeline(
            name='mean_displacement_calculation',
            inputs=[DatasetSpec('t1_1_motion_mats', directory_format),
                    DatasetSpec('ref_motion_mats', directory_format),
                    DatasetSpec('ute_motion_mats', directory_format),
                    DatasetSpec('fm_motion_mats', directory_format),
                    DatasetSpec('epi1_motion_mats', directory_format),
                    DatasetSpec('epi2_motion_mats', directory_format),
                    DatasetSpec('epi3_motion_mats', directory_format),
                    DatasetSpec('epi4_motion_mats', directory_format),
                    DatasetSpec('epi5_motion_mats', directory_format),
                    DatasetSpec('epi6_motion_mats', directory_format),
                    DatasetSpec('epi7_motion_mats', directory_format),
                    DatasetSpec('epi8_motion_mats', directory_format),
                    DatasetSpec('asl_motion_mats', directory_format),
                    DatasetSpec('epi1_ref_brain', nifti_gz_format),
                    FieldSpec('ref_tr', float),
                    FieldSpec('ref_start_time', str),
                    FieldSpec('ref_real_duration', str),
                    FieldSpec('t1_1_tr', float),
                    FieldSpec('t1_1_start_time', str),
                    FieldSpec('t1_1_real_duration', str),
                    FieldSpec('ute_tr', float),
                    FieldSpec('ute_start_time', str),
                    FieldSpec('ute_real_duration', str),
                    FieldSpec('fm_tr', float),
                    FieldSpec('fm_start_time', str),
                    FieldSpec('fm_real_duration', str),
                    FieldSpec('epi1_tr', float),
                    FieldSpec('epi1_start_time', str),
                    FieldSpec('epi1_real_duration', str),
                    FieldSpec('epi2_tr', float),
                    FieldSpec('epi2_start_time', str),
                    FieldSpec('epi2_real_duration', str),
                    FieldSpec('epi3_tr', float),
                    FieldSpec('epi3_start_time', str),
                    FieldSpec('epi3_real_duration', str),
                    FieldSpec('epi4_tr', float),
                    FieldSpec('epi4_start_time', str),
                    FieldSpec('epi4_real_duration', str),
                    FieldSpec('epi5_tr', float),
                    FieldSpec('epi5_start_time', str),
                    FieldSpec('epi5_real_duration', str),
                    FieldSpec('epi6_tr', float),
                    FieldSpec('epi6_start_time', str),
                    FieldSpec('epi6_real_duration', str),
                    FieldSpec('epi7_tr', float),
                    FieldSpec('epi7_start_time', str),
                    FieldSpec('epi7_real_duration', str),
                    FieldSpec('epi8_tr', float),
                    FieldSpec('epi8_start_time', str),
                    FieldSpec('epi8_real_duration', str),
                    FieldSpec('asl_tr', float),
                    FieldSpec('asl_start_time', str),
                    FieldSpec('asl_real_duration', str)],
            outputs=[DatasetSpec('mean_displacement', text_format),
                     DatasetSpec('mean_displacement_rc', text_format),
                     DatasetSpec('mean_displacement_consecutive', text_format),
                     DatasetSpec('start_times', text_format),
                     DatasetSpec('motion_par_rc', text_format),
                     DatasetSpec('offset_indexes', text_format)],
            description=("Calculate the mean displacement between each motion"
                         " matrix and a reference."),
            default_options={},
            version=1,
            citations=[fsl_cite],
            options=options)

        merge_ref = pipeline.create_node(merge_lists(4), name='merge_ref')
        pipeline.connect_input('ref_motion_mats', merge_ref, 'in1')
        pipeline.connect_input('ref_start_time', merge_ref, 'in2')
        pipeline.connect_input('ref_real_duration', merge_ref, 'in3')
        pipeline.connect_input('ref_tr', merge_ref, 'in4')

        merge_t1 = pipeline.create_node(merge_lists(4), name='merge_t1')
        pipeline.connect_input('t1_1_motion_mats', merge_t1, 'in1')
        pipeline.connect_input('t1_1_start_time', merge_t1, 'in2')
        pipeline.connect_input('t1_1_real_duration', merge_t1, 'in3')
        pipeline.connect_input('t1_1_tr', merge_t1, 'in4')

        merge_ute = pipeline.create_node(merge_lists(4), name='merge_ute')
        pipeline.connect_input('ute_motion_mats', merge_ute, 'in1')
        pipeline.connect_input('ute_start_time', merge_ute, 'in2')
        pipeline.connect_input('ute_real_duration', merge_ute, 'in3')
        pipeline.connect_input('ute_tr', merge_ute, 'in4')

        merge_fm = pipeline.create_node(merge_lists(4), name='merge_fm')
        pipeline.connect_input('fm_motion_mats', merge_fm, 'in1')
        pipeline.connect_input('fm_start_time', merge_fm, 'in2')
        pipeline.connect_input('fm_real_duration', merge_fm, 'in3')
        pipeline.connect_input('fm_tr', merge_fm, 'in4')

        merge_epi1 = pipeline.create_node(merge_lists(4), name='merge_epi1')
        pipeline.connect_input('epi1_motion_mats', merge_epi1, 'in1')
        pipeline.connect_input('epi1_start_time', merge_epi1, 'in2')
        pipeline.connect_input('epi1_real_duration', merge_epi1, 'in3')
        pipeline.connect_input('epi1_tr', merge_epi1, 'in4')

        merge_epi2 = pipeline.create_node(merge_lists(4), name='merge_epi2')
        pipeline.connect_input('epi2_motion_mats', merge_epi2, 'in1')
        pipeline.connect_input('epi2_start_time', merge_epi2, 'in2')
        pipeline.connect_input('epi2_real_duration', merge_epi2, 'in3')
        pipeline.connect_input('epi2_tr', merge_epi2, 'in4')

        merge_epi3 = pipeline.create_node(merge_lists(4), name='merge_epi3')
        pipeline.connect_input('epi3_motion_mats', merge_epi3, 'in1')
        pipeline.connect_input('epi3_start_time', merge_epi3, 'in2')
        pipeline.connect_input('epi3_real_duration', merge_epi3, 'in3')
        pipeline.connect_input('epi3_tr', merge_epi3, 'in4')

        merge_epi4 = pipeline.create_node(merge_lists(4), name='merge_epi4')
        pipeline.connect_input('epi4_motion_mats', merge_epi4, 'in1')
        pipeline.connect_input('epi4_start_time', merge_epi4, 'in2')
        pipeline.connect_input('epi4_real_duration', merge_epi4, 'in3')
        pipeline.connect_input('epi4_tr', merge_epi4, 'in4')

        merge_epi5 = pipeline.create_node(merge_lists(4), name='merge_epi5')
        pipeline.connect_input('epi5_motion_mats', merge_epi5, 'in1')
        pipeline.connect_input('epi5_start_time', merge_epi5, 'in2')
        pipeline.connect_input('epi5_real_duration', merge_epi5, 'in3')
        pipeline.connect_input('epi5_tr', merge_epi5, 'in4')

        merge_epi6 = pipeline.create_node(merge_lists(4), name='merge_epi6')
        pipeline.connect_input('epi6_motion_mats', merge_epi6, 'in1')
        pipeline.connect_input('epi6_start_time', merge_epi6, 'in2')
        pipeline.connect_input('epi6_real_duration', merge_epi6, 'in3')
        pipeline.connect_input('epi6_tr', merge_epi6, 'in4')

        merge_epi7 = pipeline.create_node(merge_lists(4), name='merge_epi7')
        pipeline.connect_input('epi7_motion_mats', merge_epi7, 'in1')
        pipeline.connect_input('epi7_start_time', merge_epi7, 'in2')
        pipeline.connect_input('epi7_real_duration', merge_epi7, 'in3')
        pipeline.connect_input('epi7_tr', merge_epi7, 'in4')

        merge_epi8 = pipeline.create_node(merge_lists(4), name='merge_epi8')
        pipeline.connect_input('epi8_motion_mats', merge_epi8, 'in1')
        pipeline.connect_input('epi8_start_time', merge_epi8, 'in2')
        pipeline.connect_input('epi8_real_duration', merge_epi8, 'in3')
        pipeline.connect_input('epi8_tr', merge_epi8, 'in4')

        merge_asl = pipeline.create_node(merge_lists(4), name='merge_asl')
        pipeline.connect_input('asl_motion_mats', merge_asl, 'in1')
        pipeline.connect_input('asl_start_time', merge_asl, 'in2')
        pipeline.connect_input('asl_real_duration', merge_asl, 'in3')
        pipeline.connect_input('asl_tr', merge_asl, 'in4')

        merge_scans = pipeline.create_node(merge_lists(13), name='merge_scans')
        merge_scans.inputs.no_flatten = True
        pipeline.connect(merge_epi1, 'out', merge_scans, 'in1')
        pipeline.connect(merge_epi2, 'out', merge_scans, 'in2')
        pipeline.connect(merge_epi3, 'out', merge_scans, 'in3')
        pipeline.connect(merge_epi4, 'out', merge_scans, 'in4')
        pipeline.connect(merge_epi5, 'out', merge_scans, 'in5')
        pipeline.connect(merge_epi6, 'out', merge_scans, 'in6')
        pipeline.connect(merge_epi7, 'out', merge_scans, 'in7')
        pipeline.connect(merge_epi8, 'out', merge_scans, 'in8')
        pipeline.connect(merge_asl, 'out', merge_scans, 'in9')
        pipeline.connect(merge_t1, 'out', merge_scans, 'in10')
        pipeline.connect(merge_ute, 'out', merge_scans, 'in11')
        pipeline.connect(merge_fm, 'out', merge_scans, 'in12')
        pipeline.connect(merge_ref, 'out', merge_scans, 'in13')

        md = pipeline.create_node(MeanDisplacementCalculation(),
                                  name='scan_time_info')
        pipeline.connect(merge_scans, 'out', md, 'list_inputs')
        pipeline.connect_input('epi1_ref_brain', md, 'reference')
        pipeline.connect_output('mean_displacement', md, 'mean_displacement')
        pipeline.connect_output(
            'mean_displacement_rc', md, 'mean_displacement_rc')
        pipeline.connect_output(
            'mean_displacement_consecutive', md,
            'mean_displacement_consecutive')
        pipeline.connect_output('start_times', md, 'start_times')
        pipeline.connect_output('motion_par_rc', md, 'motion_parameters')
        pipeline.connect_output('offset_indexes', md, 'offset_indexes')
        pipeline.assert_connected()
        return pipeline

    def motion_framing_pipeline(self, **options):

        pipeline = self.create_pipeline(
            name='motion_framing',
            inputs=[DatasetSpec('mean_displacement', text_format),
                    DatasetSpec('mean_displacement_consecutive', text_format),
                    DatasetSpec('start_times', text_format)],
            outputs=[DatasetSpec('frame_start_times', text_format)],
            description=("Calculate when the head movement exceeded a "
                         "predefined threshold (default 2mm)."),
            default_options={'th': 2.0, 'temporal_th': 30.0},
            version=1,
            citations=[fsl_cite],
            options=options)

        framing = pipeline.create_node(MotionFraming(), name='motion_framing')
        framing.inputs.motion_threshold = pipeline.option('th')
        framing.inputs.temporal_threshold = pipeline.option('temporal_th')
        pipeline.connect_input('mean_displacement', framing,
                               'mean_displacement')
        pipeline.connect_input('mean_displacement_consecutive', framing,
                               'mean_displacement_consec')
        pipeline.connect_input('start_times', framing, 'start_times')
        pipeline.connect_output('frame_start_times', framing,
                                'frame_start_times')
        pipeline.assert_connected()
        return pipeline

    _sub_study_specs = set_specs(
        SubStudySpec('ref', MotionReferenceStudy, {
            'reference': 'primary',
            'ref_ped': 'ped',
            'ref_pe_angle': 'pe_angle',
            'ref_tr': 'tr',
            'ref_real_duration': 'real_duration',
            'ref_tot_duration': 'tot_duration',
            'ref_start_time': 'start_time',
            'ref_dcm_info': 'dcm_info',
            'ref_motion_mats': 'ref_motion_mats'}),
        SubStudySpec('fm', CoregisteredT2Study, {
            'fm': 't2',
            'fm_nifti': 't2_nifti',
            'fm_reg': 't2_reg',
            'fm_reg_mat': 't2_reg_mat',
            'fm_qformed': 't2_qformed',
            'fm_qform_mat': 't2_qform_mat',
            'fm_reference': 'reference',
            'fm_brain': 't2_brain',
            'fm_brain_mask': 't2_brain_mask',
            'fm_preproc': 't2_preproc',
            'fm_ref_preproc': 'ref_preproc',
            'fm_ref_brain': 'ref_brain',
            'fm_ref_brain_mask': 'ref_brain_mask',
            'fm_ped': 't2_ped',
            'fm_pe_angle': 't2_pe_angle',
            'fm_tr': 't2_tr',
            'fm_real_duration': 't2_real_duration',
            'fm_tot_duration': 't2_tot_duration',
            'fm_start_time': 't2_start_time',
            'fm_dcm_info': 't2_dcm_info',
            'fm_motion_mats': 't2_motion_mats'}),
        SubStudySpec('ute', CoregisteredT1Study, {
            'ute': 't1',
            'ute_nifti': 't1_nifti',
            'ute_reg': 't1_reg',
            'ute_reg_mat': 't1_reg_mat',
            'ute_qformed': 't1_qformed',
            'ute_qform_mat': 't1_qform_mat',
            'ute_reference': 'reference',
            'ute_brain': 't1_brain',
            'ute_brain_mask': 't1_brain_mask',
            'ute_preproc': 't1_preproc',
            'ute_ref_preproc': 'ref_preproc',
            'ute_ref_brain': 'ref_brain',
            'ute_ref_brain_mask': 'ref_brain_mask',
            'ute_ped': 't1_ped',
            'ute_pe_angle': 't1_pe_angle',
            'ute_tr': 't1_tr',
            'ute_real_duration': 't1_real_duration',
            'ute_tot_duration': 't1_tot_duration',
            'ute_start_time': 't1_start_time',
            'ute_dcm_info': 't1_dcm_info',
            'ute_motion_mats': 't1_motion_mats'}),
        SubStudySpec('t1_1', CoregisteredT1Study, {
            't1_1': 't1',
            't1_1_nifti': 't1_nifti',
            't1_1_reg': 't1_reg',
            't1_1_reg_mat': 't1_reg_mat',
            't1_1_qformed': 't1_qformed',
            't1_1_qform_mat': 't1_qform_mat',
            't1_1_reference': 'reference',
            't1_1_brain': 't1_brain',
            't1_1_brain_mask': 't1_brain_mask',
            't1_1_preproc': 't1_preproc',
            't1_1_ref_preproc': 'ref_preproc',
            't1_1_ref_brain': 'ref_brain',
            't1_1_ref_brain_mask': 'ref_brain_mask',
            't1_1_ped': 't1_ped',
            't1_1_pe_angle': 't1_pe_angle',
            't1_1_tr': 't1_tr',
            't1_1_real_duration': 't1_real_duration',
            't1_1_tot_duration': 't1_tot_duration',
            't1_1_start_time': 't1_start_time',
            't1_1_dcm_info': 't1_dcm_info',
            't1_1_motion_mats': 't1_motion_mats'}),
        SubStudySpec('epi1', CoregisteredEPIStudy, {
            'epi1': 'epi',
            'epi1_nifti': 'epi_nifti',
            'epi1_epireg_mat': 'epi_epireg_mat',
            'epi1_epireg': 'epi_epireg',
            'epi1_qform_mat': 'epi_qform_mat',
            'epi1_qformed': 'epi_qformed',
            'epi1_moco_mat': 'epi_moco_mat',
            'epi1_moco': 'epi_moco',
            'epi1_moco_par': 'epi_moco_par',
            'epi1_motion_mats': 'epi_motion_mats',
            'epi1_preproc': 'epi_preproc',
            'epi1_ref_brain': 'ref_brain',
            'epi1_ref_preproc': 'ref_preproc',
            'epi1_ref_brain_mask': 'ref_brain_mask',
            'epi1_ref_wmseg': 'ref_wmseg',
            'epi1_reference': 'reference',
            'epi1_brain': 'epi_brain',
            'epi1_brain_mask': 'epi_brain_mask',
            'epi1_ped': 'epi_ped',
            'epi1_pe_angle': 'epi_pe_angle',
            'epi1_tr': 'epi_tr',
            'epi1_real_duration': 'epi_real_duration',
            'epi1_tot_duration': 'epi_tot_duration',
            'epi1_start_time': 'epi_start_time',
            'epi1_dcm_info': 'epi_dcm_info'}),
        SubStudySpec('epi2', CoregisteredEPIStudy, {
            'epi2': 'epi',
            'epi2_nifti': 'epi_nifti',
            'epi2_epireg_mat': 'epi_epireg_mat',
            'epi2_epireg': 'epi_epireg',
            'epi2_qform_mat': 'epi_qform_mat',
            'epi2_qformed': 'epi_qformed',
            'epi2_moco_mat': 'epi_moco_mat',
            'epi2_moco': 'epi_moco',
            'epi2_moco_par': 'epi_moco_par',
            'epi2_motion_mats': 'epi_motion_mats',
            'epi2_preproc': 'epi_preproc',
            'epi2_ref_brain': 'ref_brain',
            'epi2_ref_preproc': 'ref_preproc',
            'epi2_ref_brain_mask': 'ref_brain_mask',
            'epi2_ref_wmseg': 'ref_wmseg',
            'epi2_reference': 'reference',
            'epi2_brain': 'epi_brain',
            'epi2_brain_mask': 'epi_brain_mask',
            'epi2_ped': 'epi_ped',
            'epi2_pe_angle': 'epi_pe_angle',
            'epi2_tr': 'epi_tr',
            'epi2_real_duration': 'epi_real_duration',
            'epi2_tot_duration': 'epi_tot_duration',
            'epi2_start_time': 'epi_start_time',
            'epi2_dcm_info': 'epi_dcm_info'}),
        SubStudySpec('epi3', CoregisteredEPIStudy, {
            'epi3': 'epi',
            'epi3_nifti': 'epi_nifti',
            'epi3_epireg_mat': 'epi_epireg_mat',
            'epi3_epireg': 'epi_epireg',
            'epi3_qform_mat': 'epi_qform_mat',
            'epi3_qformed': 'epi_qformed',
            'epi3_moco_mat': 'epi_moco_mat',
            'epi3_moco': 'epi_moco',
            'epi3_moco_par': 'epi_moco_par',
            'epi3_motion_mats': 'epi_motion_mats',
            'epi3_preproc': 'epi_preproc',
            'epi3_ref_brain': 'ref_brain',
            'epi3_ref_preproc': 'ref_preproc',
            'epi3_ref_brain_mask': 'ref_brain_mask',
            'epi3_ref_wmseg': 'ref_wmseg',
            'epi3_reference': 'reference',
            'epi3_brain': 'epi_brain',
            'epi3_brain_mask': 'epi_brain_mask',
            'epi3_ped': 'epi_ped',
            'epi3_pe_angle': 'epi_pe_angle',
            'epi3_tr': 'epi_tr',
            'epi3_real_duration': 'epi_real_duration',
            'epi3_tot_duration': 'epi_tot_duration',
            'epi3_start_time': 'epi_start_time',
            'epi3_dcm_info': 'epi_dcm_info'}),
        SubStudySpec('epi4', CoregisteredEPIStudy, {
            'epi4': 'epi',
            'epi4_nifti': 'epi_nifti',
            'epi4_epireg_mat': 'epi_epireg_mat',
            'epi4_epireg': 'epi_epireg',
            'epi4_qform_mat': 'epi_qform_mat',
            'epi4_qformed': 'epi_qformed',
            'epi4_moco_mat': 'epi_moco_mat',
            'epi4_moco': 'epi_moco',
            'epi4_moco_par': 'epi_moco_par',
            'epi4_motion_mats': 'epi_motion_mats',
            'epi4_preproc': 'epi_preproc',
            'epi4_ref_brain': 'ref_brain',
            'epi4_ref_preproc': 'ref_preproc',
            'epi4_ref_brain_mask': 'ref_brain_mask',
            'epi4_ref_wmseg': 'ref_wmseg',
            'epi4_reference': 'reference',
            'epi4_brain': 'epi_brain',
            'epi4_brain_mask': 'epi_brain_mask',
            'epi4_ped': 'epi_ped',
            'epi4_pe_angle': 'epi_pe_angle',
            'epi4_tr': 'epi_tr',
            'epi4_real_duration': 'epi_real_duration',
            'epi4_tot_duration': 'epi_tot_duration',
            'epi4_start_time': 'epi_start_time',
            'epi4_dcm_info': 'epi_dcm_info'}),
        SubStudySpec('epi5', CoregisteredEPIStudy, {
            'epi5': 'epi',
            'epi5_nifti': 'epi_nifti',
            'epi5_epireg_mat': 'epi_epireg_mat',
            'epi5_epireg': 'epi_epireg',
            'epi5_qform_mat': 'epi_qform_mat',
            'epi5_qformed': 'epi_qformed',
            'epi5_moco_mat': 'epi_moco_mat',
            'epi5_moco': 'epi_moco',
            'epi5_moco_par': 'epi_moco_par',
            'epi5_motion_mats': 'epi_motion_mats',
            'epi5_preproc': 'epi_preproc',
            'epi5_ref_brain': 'ref_brain',
            'epi5_ref_preproc': 'ref_preproc',
            'epi5_ref_brain_mask': 'ref_brain_mask',
            'epi5_ref_wmseg': 'ref_wmseg',
            'epi5_reference': 'reference',
            'epi5_brain': 'epi_brain',
            'epi5_brain_mask': 'epi_brain_mask',
            'epi5_ped': 'epi_ped',
            'epi5_pe_angle': 'epi_pe_angle',
            'epi5_tr': 'epi_tr',
            'epi5_real_duration': 'epi_real_duration',
            'epi5_tot_duration': 'epi_tot_duration',
            'epi5_start_time': 'epi_start_time',
            'epi5_dcm_info': 'epi_dcm_info'}),
        SubStudySpec('epi6', CoregisteredEPIStudy, {
            'epi6': 'epi',
            'epi6_nifti': 'epi_nifti',
            'epi6_epireg_mat': 'epi_epireg_mat',
            'epi6_epireg': 'epi_epireg',
            'epi6_qform_mat': 'epi_qform_mat',
            'epi6_qformed': 'epi_qformed',
            'epi6_moco_mat': 'epi_moco_mat',
            'epi6_moco': 'epi_moco',
            'epi6_moco_par': 'epi_moco_par',
            'epi6_motion_mats': 'epi_motion_mats',
            'epi6_preproc': 'epi_preproc',
            'epi6_ref_brain': 'ref_brain',
            'epi6_ref_preproc': 'ref_preproc',
            'epi6_ref_brain_mask': 'ref_brain_mask',
            'epi6_ref_wmseg': 'ref_wmseg',
            'epi6_reference': 'reference',
            'epi6_brain': 'epi_brain',
            'epi6_brain_mask': 'epi_brain_mask',
            'epi6_ped': 'epi_ped',
            'epi6_pe_angle': 'epi_pe_angle',
            'epi6_tr': 'epi_tr',
            'epi6_real_duration': 'epi_real_duration',
            'epi6_tot_duration': 'epi_tot_duration',
            'epi6_start_time': 'epi_start_time',
            'epi6_dcm_info': 'epi_dcm_info'}),
        SubStudySpec('epi7', CoregisteredEPIStudy, {
            'epi7': 'epi',
            'epi7_nifti': 'epi_nifti',
            'epi7_epireg_mat': 'epi_epireg_mat',
            'epi7_epireg': 'epi_epireg',
            'epi7_qform_mat': 'epi_qform_mat',
            'epi7_qformed': 'epi_qformed',
            'epi7_moco_mat': 'epi_moco_mat',
            'epi7_moco': 'epi_moco',
            'epi7_moco_par': 'epi_moco_par',
            'epi7_motion_mats': 'epi_motion_mats',
            'epi7_preproc': 'epi_preproc',
            'epi7_ref_brain': 'ref_brain',
            'epi7_ref_preproc': 'ref_preproc',
            'epi7_ref_brain_mask': 'ref_brain_mask',
            'epi7_ref_wmseg': 'ref_wmseg',
            'epi7_reference': 'reference',
            'epi7_brain': 'epi_brain',
            'epi7_brain_mask': 'epi_brain_mask',
            'epi7_ped': 'epi_ped',
            'epi7_pe_angle': 'epi_pe_angle',
            'epi7_tr': 'epi_tr',
            'epi7_real_duration': 'epi_real_duration',
            'epi7_tot_duration': 'epi_tot_duration',
            'epi7_start_time': 'epi_start_time',
            'epi7_dcm_info': 'epi_dcm_info'}),
        SubStudySpec('epi8', CoregisteredEPIStudy, {
            'epi8': 'epi',
            'epi8_nifti': 'epi_nifti',
            'epi8_epireg_mat': 'epi_epireg_mat',
            'epi8_epireg': 'epi_epireg',
            'epi8_qform_mat': 'epi_qform_mat',
            'epi8_qformed': 'epi_qformed',
            'epi8_moco_mat': 'epi_moco_mat',
            'epi8_moco': 'epi_moco',
            'epi8_moco_par': 'epi_moco_par',
            'epi8_motion_mats': 'epi_motion_mats',
            'epi8_preproc': 'epi_preproc',
            'epi8_ref_brain': 'ref_brain',
            'epi8_ref_preproc': 'ref_preproc',
            'epi8_ref_brain_mask': 'ref_brain_mask',
            'epi8_ref_wmseg': 'ref_wmseg',
            'epi8_reference': 'reference',
            'epi8_brain': 'epi_brain',
            'epi8_brain_mask': 'epi_brain_mask',
            'epi8_ped': 'epi_ped',
            'epi8_pe_angle': 'epi_pe_angle',
            'epi8_tr': 'epi_tr',
            'epi8_real_duration': 'epi_real_duration',
            'epi8_tot_duration': 'epi_tot_duration',
            'epi8_start_time': 'epi_start_time',
            'epi8_dcm_info': 'epi_dcm_info'}),
        SubStudySpec('asl', CoregisteredEPIStudy, {
            'asl': 'epi',
            'asl_nifti': 'epi_nifti',
            'asl_epireg_mat': 'epi_epireg_mat',
            'asl_epireg': 'epi_epireg',
            'asl_qform_mat': 'epi_qform_mat',
            'asl_qformed': 'epi_qformed',
            'asl_moco_mat': 'epi_moco_mat',
            'asl_moco': 'epi_moco',
            'asl_moco_par': 'epi_moco_par',
            'asl_motion_mats': 'epi_motion_mats',
            'asl_preproc': 'epi_preproc',
            'asl_ref_brain': 'ref_brain',
            'asl_ref_preproc': 'ref_preproc',
            'asl_ref_brain_mask': 'ref_brain_mask',
            'asl_ref_wmseg': 'ref_wmseg',
            'asl_reference': 'reference',
            'asl_brain': 'epi_brain',
            'asl_brain_mask': 'epi_brain_mask',
            'asl_ped': 'epi_ped',
            'asl_pe_angle': 'epi_pe_angle',
            'asl_tr': 'epi_tr',
            'asl_real_duration': 'epi_real_duration',
            'asl_tot_duration': 'epi_tot_duration',
            'asl_start_time': 'epi_start_time',
            'asl_dcm_info': 'epi_dcm_info'}))

    _data_specs = set_specs([
        DatasetSpec('reference', dicom_format),
        DatasetSpec('ref_motion_mats', directory_format,
                    'ref_dcm_info_pipeline'),
        DatasetSpec('ref_dcm_info', text_format,
                    'ref_dcm_info_pipeline'),
        FieldSpec('ref_ped', str, 'ref_dcm_info_pipeline'),
        FieldSpec('ref_pe_angle', str, 'ref_dcm_info_pipeline'),
        FieldSpec('ref_tr', float, 'ref_dcm_info_pipeline'),
        FieldSpec('ref_start_time', str, 'ref_dcm_info_pipeline'),
        FieldSpec('ref_real_duration', str, 'ref_dcm_info_pipeline'),
        FieldSpec('ref_tot_duration', str, 'ref_dcm_info_pipeline'),
        DatasetSpec('fm', dicom_format),
        DatasetSpec('fm_reference', nifti_gz_format),
        DatasetSpec('fm_nifti', nifti_gz_format, 'fm_dcm2nii_pipeline'),
        DatasetSpec('fm_reg', nifti_gz_format,
                    'fm_rigid_registration_pipeline'),
        DatasetSpec('fm_reg_mat', text_matrix_format,
                    'fm_rigid_registration_pipeline'),
        DatasetSpec('fm_qformed', nifti_gz_format,
                    'fm_qform_transform_pipeline'),
        DatasetSpec('fm_qform_mat', text_matrix_format,
                    'fm_qform_transform_pipeline'),
        DatasetSpec('fm_brain', nifti_gz_format, 'fm_bet_pipeline'),
        DatasetSpec('fm_brain_mask', nifti_gz_format, 'fm_bet_pipeline'),
        DatasetSpec('fm_preproc', nifti_gz_format,
                    'fm_basic_preproc_pipeline'),
        DatasetSpec('fm_ref_preproc', nifti_gz_format,
                    'fm_ref_basic_preproc_pipeline'),
        DatasetSpec('fm_ref_brain', nifti_gz_format, 'fm_ref_bet_pipeline'),
        DatasetSpec('fm_ref_brain_mask', nifti_gz_format,
                    'fm_ref_bet_pipeline'),
        DatasetSpec('fm_motion_mats', directory_format,
                    'fm_motion_mat_pipeline'),
        DatasetSpec('fm_dcm_info', text_format,
                    'fm_dcm_info_pipeline'),
        FieldSpec('fm_ped', str, 'fm_dcm_info_pipeline'),
        FieldSpec('fm_pe_angle', str, 'fm_dcm_info_pipeline'),
        FieldSpec('fm_tr', float, 'fm_dcm_info_pipeline'),
        FieldSpec('fm_start_time', str, 'fm_dcm_info_pipeline'),
        FieldSpec('fm_real_duration', str, 'fm_dcm_info_pipeline'),
        FieldSpec('fm_tot_duration', str, 'fm_dcm_info_pipeline'),
        DatasetSpec('ute', dicom_format),
        DatasetSpec('ute_reference', nifti_gz_format),
        DatasetSpec('ute_nifti', nifti_gz_format, 'ute_dcm2nii_pipeline'),
        DatasetSpec('ute_reg', nifti_gz_format,
                    'ute_rigid_registration_pipeline'),
        DatasetSpec('ute_reg_mat', text_matrix_format,
                    'ute_rigid_registration_pipeline'),
        DatasetSpec('ute_qformed', nifti_gz_format,
                    'ute_qform_transform_pipeline'),
        DatasetSpec('ute_qform_mat', text_matrix_format,
                    'ute_qform_transform_pipeline'),
        DatasetSpec('ute_brain', nifti_gz_format, 'ute_bet_pipeline'),
        DatasetSpec('ute_brain_mask', nifti_gz_format, 'ute_bet_pipeline'),
        DatasetSpec('ute_preproc', nifti_gz_format,
                    'ute_basic_preproc_pipeline'),
        DatasetSpec('ute_ref_preproc', nifti_gz_format,
                    'ute_ref_basic_preproc_pipeline'),
        DatasetSpec('ute_ref_brain', nifti_gz_format, 'ute_ref_bet_pipeline'),
        DatasetSpec('ute_ref_brain_mask', nifti_gz_format,
                    'ute_ref_bet_pipeline'),
        DatasetSpec('ute_motion_mats', directory_format,
                    'ute_motion_mat_pipeline'),
        DatasetSpec('ute_dcm_info', text_format,
                    'ute_dcm_info_pipeline'),
        FieldSpec('ute_ped', str, 'ute_dcm_info_pipeline'),
        FieldSpec('ute_pe_angle', str, 'ute_dcm_info_pipeline'),
        FieldSpec('ute_tr', float, 'ute_dcm_info_pipeline'),
        FieldSpec('ute_start_time', str, 'ute_dcm_info_pipeline'),
        FieldSpec('ute_real_duration', str, 'ute_dcm_info_pipeline'),
        FieldSpec('ute_tot_duration', str, 'ute_dcm_info_pipeline'),
        DatasetSpec('t1_1', dicom_format),
        DatasetSpec('t1_1_reference', nifti_gz_format),
        DatasetSpec('t1_1_nifti', nifti_gz_format, 't1_dcm2nii_pipeline'),
        DatasetSpec('t1_1_reg', nifti_gz_format,
                    't1_rigid_registration_pipeline'),
        DatasetSpec('t1_1_reg_mat', text_matrix_format,
                    't1_rigid_registration_pipeline'),
        DatasetSpec('t1_1_qformed', nifti_gz_format,
                    't1_qform_transform_pipeline'),
        DatasetSpec('t1_1_qform_mat', text_matrix_format,
                    't1_qform_transform_pipeline'),
        DatasetSpec('t1_1_brain', nifti_gz_format, 't1_bet_pipeline'),
        DatasetSpec('t1_1_brain_mask', nifti_gz_format, 't1_bet_pipeline'),
        DatasetSpec('t1_1_preproc', nifti_gz_format,
                    't1_basic_preproc_pipeline'),
        DatasetSpec('t1_1_ref_preproc', nifti_gz_format,
                    't1_ref_basic_preproc_pipeline'),
        DatasetSpec('t1_1_ref_brain', nifti_gz_format, 't1_ref_bet_pipeline'),
        DatasetSpec('t1_1_ref_brain_mask', nifti_gz_format,
                    't1_ref_bet_pipeline'),
        DatasetSpec('t1_1_motion_mats', directory_format,
                    't1_motion_mat_pipeline'),
        DatasetSpec('t1_1_dcm_info', text_format,
                    't1_dcm_info_pipeline'),
        FieldSpec('t1_1_ped', str, 't1_dcm_info_pipeline'),
        FieldSpec('t1_1_pe_angle', str, 't1_dcm_info_pipeline'),
        FieldSpec('t1_1_tr', float, 't1_dcm_info_pipeline'),
        FieldSpec('t1_1_start_time', str, 't1_dcm_info_pipeline'),
        FieldSpec('t1_1_real_duration', str, 't1_dcm_info_pipeline'),
        FieldSpec('t1_1_tot_duration', str, 't1_dcm_info_pipeline'),
        DatasetSpec('epi1', dicom_format),
        DatasetSpec('epi1_reference', nifti_gz_format),
        DatasetSpec('epi1_nifti', nifti_gz_format, 'epi1_dcm2nii_pipeline'),
        DatasetSpec('epi1_preproc', nifti_gz_format,
                    'epi1_basic_preproc_pipeline'),
        DatasetSpec('epi1_brain', nifti_gz_format,
                    'epi1_bet_pipeline'),
        DatasetSpec('epi1_brain_mask', nifti_gz_format,
                    'epi1_bet_pipeline'),
        DatasetSpec('epi1_qformed', nifti_gz_format,
                    'epi1_qform_transform_pipeline'),
        DatasetSpec('epi1_qform_mat', text_matrix_format,
                    'epi1_qform_transform_pipeline'),
        DatasetSpec('epi1_epireg', nifti_gz_format, 'epi1_epireg_pipeline'),
        DatasetSpec('epi1_epireg_mat', text_matrix_format,
                    'epi1_epireg_pipeline'),
        DatasetSpec('epi1_motion_mats', directory_format,
                    'epi1_motion_mat_pipeline'),
        DatasetSpec('epi1_moco', nifti_gz_format,
                    'epi1_motion_alignment_pipeline'),
        DatasetSpec('epi1_moco_mat', directory_format,
                    'epi1_motion_alignment_pipeline'),
        DatasetSpec('epi1_moco_par', par_format,
                    'epi1_motion_alignment_pipeline'),
        DatasetSpec('epi1_ref_preproc', nifti_gz_format,
                    'epi1_ref_basic_preproc_pipeline'),
        DatasetSpec('epi1_ref_brain', nifti_gz_format, 'epi1_ref_bet_pipeline'),
        DatasetSpec('epi1_ref_brain_mask', nifti_gz_format,
                    'epi1_ref_bet_pipeline'),
        DatasetSpec('epi1_ref_wmseg', nifti_gz_format,
                    'epi1_ref_segmentation_pipeline'),
        DatasetSpec('epi1_dcm_info', text_format,
                    'epi1_dcm_info_pipeline'),
        FieldSpec('epi1_ped', str, 'epi1_dcm_info_pipeline'),
        FieldSpec('epi1_pe_angle', str, 'epi1_dcm_info_pipeline'),
        FieldSpec('epi1_tr', float, 'epi1_dcm_info_pipeline'),
        FieldSpec('epi1_start_time', str, 'epi1_dcm_info_pipeline'),
        FieldSpec('epi1_real_duration', str, 'epi1_dcm_info_pipeline'),
        FieldSpec('epi1_tot_duration', str, 'epi1_dcm_info_pipeline'),
        DatasetSpec('epi2', dicom_format),
        DatasetSpec('epi2_nifti', nifti_gz_format, 'epi2_dcm2nii_pipeline'),
        DatasetSpec('epi2_reference', nifti_gz_format),
        DatasetSpec('epi2_preproc', nifti_gz_format,
                    'epi2_basic_preproc_pipeline'),
        DatasetSpec('epi2_qformed', nifti_gz_format,
                    'epi2_qform_transform_pipeline'),
        DatasetSpec('epi2_qform_mat', text_matrix_format,
                    'epi2_qform_transform_pipeline'),
        DatasetSpec('epi2_epireg', nifti_gz_format, 'epi2_epireg_pipeline'),
        DatasetSpec('epi2_epireg_mat', text_matrix_format,
                    'epi2_epireg_pipeline'),
        DatasetSpec('epi2_motion_mats', directory_format,
                    'epi2_motion_mat_pipeline'),
        DatasetSpec('epi2_moco', nifti_gz_format,
                    'epi2_motion_alignment_pipeline'),
        DatasetSpec('epi2_moco_mat', directory_format,
                    'epi2_motion_alignment_pipeline'),
        DatasetSpec('epi2_moco_par', par_format,
                    'epi2_motion_alignment_pipeline'),
        DatasetSpec('epi2_brain', nifti_gz_format,
                    'epi2_bet_pipeline'),
        DatasetSpec('epi2_brain_mask', nifti_gz_format,
                    'epi2_bet_pipeline'),
        DatasetSpec('epi2_ref_preproc', nifti_gz_format,
                    'epi2_ref_basic_preproc_pipeline'),
        DatasetSpec('epi2_ref_brain', nifti_gz_format, 'epi2_ref_bet_pipeline'),
        DatasetSpec('epi2_ref_brain_mask', nifti_gz_format,
                    'epi2_ref_bet_pipeline'),
        DatasetSpec('epi2_ref_wmseg', nifti_gz_format,
                    'epi2_ref_segmentation_pipeline'),
        DatasetSpec('epi2_dcm_info', text_format,
                    'epi2_dcm_info_pipeline'),
        FieldSpec('epi2_ped', str, 'epi2_dcm_info_pipeline'),
        FieldSpec('epi2_pe_angle', str, 'epi2_dcm_info_pipeline'),
        FieldSpec('epi2_tr', float, 'epi2_dcm_info_pipeline'),
        FieldSpec('epi2_start_time', str, 'epi2_dcm_info_pipeline'),
        FieldSpec('epi2_real_duration', str, 'epi2_dcm_info_pipeline'),
        FieldSpec('epi2_tot_duration', str, 'epi2_dcm_info_pipeline'),
        DatasetSpec('epi3', dicom_format),
        DatasetSpec('epi3_nifti', nifti_gz_format, 'epi3_dcm2nii_pipeline'),
        DatasetSpec('epi3_reference', nifti_gz_format),
        DatasetSpec('epi3_preproc', nifti_gz_format,
                    'epi3_basic_preproc_pipeline'),
        DatasetSpec('epi3_qformed', nifti_gz_format,
                    'epi3_qform_transform_pipeline'),
        DatasetSpec('epi3_qform_mat', text_matrix_format,
                    'epi3_qform_transform_pipeline'),
        DatasetSpec('epi3_epireg', nifti_gz_format, 'epi3_epireg_pipeline'),
        DatasetSpec('epi3_epireg_mat', text_matrix_format,
                    'epi3_epireg_pipeline'),
        DatasetSpec('epi3_motion_mats', directory_format,
                    'epi3_motion_mat_pipeline'),
        DatasetSpec('epi3_moco', nifti_gz_format,
                    'epi3_motion_alignment_pipeline'),
        DatasetSpec('epi3_moco_mat', directory_format,
                    'epi3_motion_alignment_pipeline'),
        DatasetSpec('epi3_moco_par', par_format,
                    'epi3_motion_alignment_pipeline'),
        DatasetSpec('epi3_brain', nifti_gz_format,
                    'epi3_bet_pipeline'),
        DatasetSpec('epi3_brain_mask', nifti_gz_format,
                    'epi3_bet_pipeline'),
        DatasetSpec('epi3_ref_preproc', nifti_gz_format,
                    'epi3_ref_basic_preproc_pipeline'),
        DatasetSpec('epi3_ref_brain', nifti_gz_format, 'epi3_ref_bet_pipeline'),
        DatasetSpec('epi3_ref_brain_mask', nifti_gz_format,
                    'epi3_ref_bet_pipeline'),
        DatasetSpec('epi3_ref_wmseg', nifti_gz_format,
                    'epi3_ref_segmentation_pipeline'),
        DatasetSpec('epi3_dcm_info', text_format,
                    'epi3_dcm_info_pipeline'),
        FieldSpec('epi3_ped', str, 'epi3_dcm_info_pipeline'),
        FieldSpec('epi3_pe_angle', str, 'epi3_dcm_info_pipeline'),
        FieldSpec('epi3_tr', float, 'epi3_dcm_info_pipeline'),
        FieldSpec('epi3_start_time', str, 'epi3_dcm_info_pipeline'),
        FieldSpec('epi3_real_duration', str, 'epi3_dcm_info_pipeline'),
        FieldSpec('epi3_tot_duration', str, 'epi3_dcm_info_pipeline'),
        DatasetSpec('epi4', dicom_format),
        DatasetSpec('epi4_nifti', nifti_gz_format, 'epi4_dcm2nii_pipeline'),
        DatasetSpec('epi4_reference', nifti_gz_format),
        DatasetSpec('epi4_preproc', nifti_gz_format,
                    'epi4_basic_preproc_pipeline'),
        DatasetSpec('epi4_qformed', nifti_gz_format,
                    'epi4_qform_transform_pipeline'),
        DatasetSpec('epi4_qform_mat', text_matrix_format,
                    'epi4_qform_transform_pipeline'),
        DatasetSpec('epi4_epireg', nifti_gz_format, 'epi4_epireg_pipeline'),
        DatasetSpec('epi4_epireg_mat', text_matrix_format,
                    'epi4_epireg_pipeline'),
        DatasetSpec('epi4_motion_mats', directory_format,
                    'epi4_motion_mat_pipeline'),
        DatasetSpec('epi4_moco', nifti_gz_format,
                    'epi4_motion_alignment_pipeline'),
        DatasetSpec('epi4_moco_mat', directory_format,
                    'epi4_motion_alignment_pipeline'),
        DatasetSpec('epi4_moco_par', par_format,
                    'epi4_motion_alignment_pipeline'),
        DatasetSpec('epi4_brain', nifti_gz_format,
                    'epi4_bet_pipeline'),
        DatasetSpec('epi4_brain_mask', nifti_gz_format,
                    'epi4_bet_pipeline'),
        DatasetSpec('epi4_ref_preproc', nifti_gz_format,
                    'epi4_ref_basic_preproc_pipeline'),
        DatasetSpec('epi4_ref_brain', nifti_gz_format, 'epi4_ref_bet_pipeline'),
        DatasetSpec('epi4_ref_brain_mask', nifti_gz_format,
                    'epi4_ref_bet_pipeline'),
        DatasetSpec('epi4_ref_wmseg', nifti_gz_format,
                    'epi4_ref_segmentation_pipeline'),
        DatasetSpec('epi4_dcm_info', text_format,
                    'epi4_dcm_info_pipeline'),
        FieldSpec('epi4_ped', str, 'epi4_dcm_info_pipeline'),
        FieldSpec('epi4_pe_angle', str, 'epi4_dcm_info_pipeline'),
        FieldSpec('epi4_tr', float, 'epi4_dcm_info_pipeline'),
        FieldSpec('epi4_start_time', str, 'epi4_dcm_info_pipeline'),
        FieldSpec('epi4_real_duration', str, 'epi4_dcm_info_pipeline'),
        FieldSpec('epi4_tot_duration', str, 'epi4_dcm_info_pipeline'),
        DatasetSpec('epi5', dicom_format),
        DatasetSpec('epi5_nifti', nifti_gz_format, 'epi5_dcm2nii_pipeline'),
        DatasetSpec('epi5_reference', nifti_gz_format),
        DatasetSpec('epi5_preproc', nifti_gz_format,
                    'epi5_basic_preproc_pipeline'),
        DatasetSpec('epi5_qformed', nifti_gz_format,
                    'epi5_qform_transform_pipeline'),
        DatasetSpec('epi5_qform_mat', text_matrix_format,
                    'epi5_qform_transform_pipeline'),
        DatasetSpec('epi5_epireg', nifti_gz_format, 'epi5_epireg_pipeline'),
        DatasetSpec('epi5_epireg_mat', text_matrix_format,
                    'epi5_epireg_pipeline'),
        DatasetSpec('epi5_motion_mats', directory_format,
                    'epi5_motion_mat_pipeline'),
        DatasetSpec('epi5_moco', nifti_gz_format,
                    'epi5_motion_alignment_pipeline'),
        DatasetSpec('epi5_moco_mat', directory_format,
                    'epi5_motion_alignment_pipeline'),
        DatasetSpec('epi5_moco_par', par_format,
                    'epi5_motion_alignment_pipeline'),
        DatasetSpec('epi5_brain', nifti_gz_format,
                    'epi5_bet_pipeline'),
        DatasetSpec('epi5_brain_mask', nifti_gz_format,
                    'epi5_bet_pipeline'),
        DatasetSpec('epi5_ref_preproc', nifti_gz_format,
                    'epi5_ref_basic_preproc_pipeline'),
        DatasetSpec('epi5_ref_brain', nifti_gz_format, 'epi5_ref_bet_pipeline'),
        DatasetSpec('epi5_ref_brain_mask', nifti_gz_format,
                    'epi5_ref_bet_pipeline'),
        DatasetSpec('epi5_ref_wmseg', nifti_gz_format,
                    'epi5_ref_segmentation_pipeline'),
        DatasetSpec('epi5_dcm_info', text_format,
                    'epi5_dcm_info_pipeline'),
        FieldSpec('epi5_ped', str, 'epi5_dcm_info_pipeline'),
        FieldSpec('epi5_pe_angle', str, 'epi5_dcm_info_pipeline'),
        FieldSpec('epi5_tr', float, 'epi5_dcm_info_pipeline'),
        FieldSpec('epi5_start_time', str, 'epi5_dcm_info_pipeline'),
        FieldSpec('epi5_real_duration', str, 'epi5_dcm_info_pipeline'),
        FieldSpec('epi5_tot_duration', str, 'epi5_dcm_info_pipeline'),
        DatasetSpec('epi6', dicom_format),
        DatasetSpec('epi6_nifti', nifti_gz_format, 'epi6_dcm2nii_pipeline'),
        DatasetSpec('epi6_reference', nifti_gz_format),
        DatasetSpec('epi6_preproc', nifti_gz_format,
                    'epi6_basic_preproc_pipeline'),
        DatasetSpec('epi6_qformed', nifti_gz_format,
                    'epi6_qform_transform_pipeline'),
        DatasetSpec('epi6_qform_mat', text_matrix_format,
                    'epi6_qform_transform_pipeline'),
        DatasetSpec('epi6_epireg', nifti_gz_format, 'epi6_epireg_pipeline'),
        DatasetSpec('epi6_epireg_mat', text_matrix_format,
                    'epi6_epireg_pipeline'),
        DatasetSpec('epi6_motion_mats', directory_format,
                    'epi6_motion_mat_pipeline'),
        DatasetSpec('epi6_moco', nifti_gz_format,
                    'epi6_motion_alignment_pipeline'),
        DatasetSpec('epi6_moco_mat', directory_format,
                    'epi6_motion_alignment_pipeline'),
        DatasetSpec('epi6_moco_par', par_format,
                    'epi6_motion_alignment_pipeline'),
        DatasetSpec('epi6_brain', nifti_gz_format,
                    'epi6_bet_pipeline'),
        DatasetSpec('epi6_brain_mask', nifti_gz_format,
                    'epi6_bet_pipeline'),
        DatasetSpec('epi6_ref_preproc', nifti_gz_format,
                    'epi6_ref_basic_preproc_pipeline'),
        DatasetSpec('epi6_ref_brain', nifti_gz_format, 'epi6_ref_bet_pipeline'),
        DatasetSpec('epi6_ref_brain_mask', nifti_gz_format,
                    'epi6_ref_bet_pipeline'),
        DatasetSpec('epi6_ref_wmseg', nifti_gz_format,
                    'epi6_ref_segmentation_pipeline'),
        DatasetSpec('epi6_dcm_info', text_format,
                    'epi6_dcm_info_pipeline'),
        FieldSpec('epi6_ped', str, 'epi6_dcm_info_pipeline'),
        FieldSpec('epi6_pe_angle', str, 'epi6_dcm_info_pipeline'),
        FieldSpec('epi6_tr', float, 'epi6_dcm_info_pipeline'),
        FieldSpec('epi6_start_time', str, 'epi6_dcm_info_pipeline'),
        FieldSpec('epi6_real_duration', str, 'epi6_dcm_info_pipeline'),
        FieldSpec('epi6_tot_duration', str, 'epi6_dcm_info_pipeline'),
        DatasetSpec('epi7', dicom_format),
        DatasetSpec('epi7_nifti', nifti_gz_format, 'epi7_dcm2nii_pipeline'),
        DatasetSpec('epi7_reference', nifti_gz_format),
        DatasetSpec('epi7_preproc', nifti_gz_format,
                    'epi7_basic_preproc_pipeline'),
        DatasetSpec('epi7_qformed', nifti_gz_format,
                    'epi7_qform_transform_pipeline'),
        DatasetSpec('epi7_qform_mat', text_matrix_format,
                    'epi7_qform_transform_pipeline'),
        DatasetSpec('epi7_epireg', nifti_gz_format, 'epi7_epireg_pipeline'),
        DatasetSpec('epi7_epireg_mat', text_matrix_format,
                    'epi7_epireg_pipeline'),
        DatasetSpec('epi7_motion_mats', directory_format,
                    'epi7_motion_mat_pipeline'),
        DatasetSpec('epi7_moco', nifti_gz_format,
                    'epi7_motion_alignment_pipeline'),
        DatasetSpec('epi7_moco_mat', directory_format,
                    'epi7_motion_alignment_pipeline'),
        DatasetSpec('epi7_moco_par', par_format,
                    'epi7_motion_alignment_pipeline'),
        DatasetSpec('epi7_brain', nifti_gz_format,
                    'epi7_bet_pipeline'),
        DatasetSpec('epi7_brain_mask', nifti_gz_format,
                    'epi7_bet_pipeline'),
        DatasetSpec('epi7_ref_preproc', nifti_gz_format,
                    'epi7_ref_basic_preproc_pipeline'),
        DatasetSpec('epi7_ref_brain', nifti_gz_format, 'epi7_ref_bet_pipeline'),
        DatasetSpec('epi7_ref_brain_mask', nifti_gz_format,
                    'epi7_ref_bet_pipeline'),
        DatasetSpec('epi7_ref_wmseg', nifti_gz_format,
                    'epi7_ref_segmentation_pipeline'),
        DatasetSpec('epi7_dcm_info', text_format,
                    'epi7_dcm_info_pipeline'),
        FieldSpec('epi7_ped', str, 'epi7_dcm_info_pipeline'),
        FieldSpec('epi7_pe_angle', str, 'epi7_dcm_info_pipeline'),
        FieldSpec('epi7_tr', float, 'epi7_dcm_info_pipeline'),
        FieldSpec('epi7_start_time', str, 'epi7_dcm_info_pipeline'),
        FieldSpec('epi7_real_duration', str, 'epi7_dcm_info_pipeline'),
        FieldSpec('epi7_tot_duration', str, 'epi7_dcm_info_pipeline'),
        DatasetSpec('epi8', dicom_format),
        DatasetSpec('epi8_nifti', nifti_gz_format, 'epi8_dcm2nii_pipeline'),
        DatasetSpec('epi8_reference', nifti_gz_format),
        DatasetSpec('epi8_preproc', nifti_gz_format,
                    'epi8_basic_preproc_pipeline'),
        DatasetSpec('epi8_qformed', nifti_gz_format,
                    'epi8_qform_transform_pipeline'),
        DatasetSpec('epi8_qform_mat', text_matrix_format,
                    'epi8_qform_transform_pipeline'),
        DatasetSpec('epi8_epireg', nifti_gz_format, 'epi8_epireg_pipeline'),
        DatasetSpec('epi8_epireg_mat', text_matrix_format,
                    'epi8_epireg_pipeline'),
        DatasetSpec('epi8_motion_mats', directory_format,
                    'epi8_motion_mat_pipeline'),
        DatasetSpec('epi8_moco', nifti_gz_format,
                    'epi8_motion_alignment_pipeline'),
        DatasetSpec('epi8_moco_mat', directory_format,
                    'epi8_motion_alignment_pipeline'),
        DatasetSpec('epi8_moco_par', par_format,
                    'epi8_motion_alignment_pipeline'),
        DatasetSpec('epi8_brain', nifti_gz_format,
                    'epi8_bet_pipeline'),
        DatasetSpec('epi8_brain_mask', nifti_gz_format,
                    'epi8_bet_pipeline'),
        DatasetSpec('epi8_ref_preproc', nifti_gz_format,
                    'epi8_ref_basic_preproc_pipeline'),
        DatasetSpec('epi8_ref_brain', nifti_gz_format, 'epi8_ref_bet_pipeline'),
        DatasetSpec('epi8_ref_brain_mask', nifti_gz_format,
                    'epi8_ref_bet_pipeline'),
        DatasetSpec('epi8_ref_wmseg', nifti_gz_format,
                    'epi8_ref_segmentation_pipeline'),
        DatasetSpec('epi8_dcm_info', text_format,
                    'epi8_dcm_info_pipeline'),
        FieldSpec('epi8_ped', str, 'epi8_dcm_info_pipeline'),
        FieldSpec('epi8_pe_angle', str, 'epi8_dcm_info_pipeline'),
        FieldSpec('epi8_tr', float, 'epi8_dcm_info_pipeline'),
        FieldSpec('epi8_start_time', str, 'epi8_dcm_info_pipeline'),
        FieldSpec('epi8_real_duration', str, 'epi8_dcm_info_pipeline'),
        FieldSpec('epi8_tot_duration', str, 'epi8_dcm_info_pipeline'),
        DatasetSpec('asl', dicom_format),
        DatasetSpec('asl_nifti', nifti_gz_format, 'asl_dcm2nii_pipeline'),
        DatasetSpec('asl_reference', nifti_gz_format),
        DatasetSpec('asl_preproc', nifti_gz_format,
                    'asl_basic_preproc_pipeline'),
        DatasetSpec('asl_qformed', nifti_gz_format,
                    'asl_qform_transform_pipeline'),
        DatasetSpec('asl_qform_mat', text_matrix_format,
                    'asl_qform_transform_pipeline'),
        DatasetSpec('asl_epireg', nifti_gz_format, 'asl_epireg_pipeline'),
        DatasetSpec('asl_epireg_mat', text_matrix_format,
                    'asl_epireg_pipeline'),
        DatasetSpec('asl_motion_mats', directory_format,
                    'asl_motion_mat_pipeline'),
        DatasetSpec('asl_moco', nifti_gz_format,
                    'asl_motion_alignment_pipeline'),
        DatasetSpec('asl_moco_mat', directory_format,
                    'asl_motion_alignment_pipeline'),
        DatasetSpec('asl_moco_par', par_format,
                    'asl_motion_alignment_pipeline'),
        DatasetSpec('asl_brain', nifti_gz_format,
                    'asl_bet_pipeline'),
        DatasetSpec('asl_brain_mask', nifti_gz_format,
                    'asl_bet_pipeline'),
        DatasetSpec('asl_ref_preproc', nifti_gz_format,
                    'asl_ref_basic_preproc_pipeline'),
        DatasetSpec('asl_ref_brain', nifti_gz_format, 'asl_ref_bet_pipeline'),
        DatasetSpec('asl_ref_brain_mask', nifti_gz_format,
                    'asl_ref_bet_pipeline'),
        DatasetSpec('asl_ref_wmseg', nifti_gz_format,
                    'asl_ref_segmentation_pipeline'),
        DatasetSpec('asl_dcm_info', text_format,
                    'asl_dcm_info_pipeline'),
        FieldSpec('asl_ped', str, 'asl_dcm_info_pipeline'),
        FieldSpec('asl_pe_angle', str, 'asl_dcm_info_pipeline'),
        FieldSpec('asl_tr', float, 'asl_dcm_info_pipeline'),
        FieldSpec('asl_start_time', str, 'asl_dcm_info_pipeline'),
        FieldSpec('asl_real_duration', str, 'asl_dcm_info_pipeline'),
        FieldSpec('asl_tot_duration', str, 'asl_dcm_info_pipeline'),
        DatasetSpec('mean_displacement', text_format,
                    'mean_displacement_pipeline'),
        DatasetSpec('mean_displacement_rc', text_format,
                    'mean_displacement_pipeline'),
        DatasetSpec('mean_displacement_consecutive', text_format,
                    'mean_displacement_pipeline'),
        DatasetSpec('start_times', text_format, 'mean_displacement_pipeline'),
        DatasetSpec('motion_par_rc', text_format, 'mean_displacement_pipeline'),
        DatasetSpec('offset_indexes', text_format, 'mean_displacement_pipeline'),
        DatasetSpec('frame_start_times', text_format,
                    'motion_framing_pipeline')])
