from nianalysis.dataset import DatasetSpec, FieldSpec
from nianalysis.data_formats import (
    nifti_gz_format, text_matrix_format, directory_format, dicom_format,
    par_format, text_format, eddy_par_format, png_format)
from mbianalysis.interfaces.custom.motion_correction import (
    MeanDisplacementCalculation, MotionFraming, PlotMeanDisplacementRC,
    AffineMatAveraging, PetCorrectionFactor, FrameAlign2Reference)
from nianalysis.citations import fsl_cite
from nianalysis.study.base import StudyMetaClass
from nianalysis.study.multi import (
    MultiStudy, SubStudySpec, MultiStudyMetaClass)
from .epi import CoregisteredEPIStudy
from .structural.t1 import CoregisteredT1Study
from .structural.t2 import CoregisteredT2Study
from nipype.interfaces.utility import Merge as merge_lists
from .base import MotionReferenceStudy
from .structural.diffusion_coreg import CoregisteredDiffusionStudy
from nianalysis.requirements import fsl509_req


class MotionDetectionStudy(MultiStudy):

    __metaclass__ = MultiStudyMetaClass

    dwi_1_main_dcm_info_pipeline = MultiStudy.translate(
        'dwi_1', 'dwi_main_dcm_info_pipeline')

    dwi_1_main_basic_preproc_pipeline = MultiStudy.translate(
        'dwi_1', 'dwi_main_dwipreproc_pipeline')

    dwi_1_main_bet_pipeline = MultiStudy.translate(
        'dwi_1', 'dwi_main_bet_pipeline')

    dwi_1_main_motion_alignment_pipeline = MultiStudy.translate(
        'dwi_1', 'dwi_main_motion_mat_pipeline')

    dwi_1_main_aff_mat_pipeline = MultiStudy.translate(
        'dwi_1', 'dwi_main_affine_mats_pipeline')

    dwi_1_main_qform_transform_pipeline = MultiStudy.translate(
        'dwi_1', 'dwi_main_qform_transform_pipeline')

    dwi_1_main_rigid_registration_pipeline = MultiStudy.translate(
        'dwi_1',
        CoregisteredDiffusionStudy.dwi_main_rigid_registration_pipeline)

    dwi_1_main_ref_bet_pipeline = MultiStudy.translate(
        'dwi_1', 'ref_bet_pipeline',
        override_default_options={'bet_method': 'optibet'})

    dwi_1_main_ref_basic_preproc_pipeline = MultiStudy.translate(
        'dwi_1', 'ref_basic_preproc_pipeline')

    dwi2ref_1_opposite_dcm_info_pipeline = MultiStudy.translate(
        'dwi_1', 'dwi2ref_opposite_dcm_info_pipeline')

    dwi2ref_1_opposite_basic_preproc_pipeline = MultiStudy.translate(
        'dwi_1', 'dwi2ref_opposite_topup_pipeline')

    dwi2ref_1_opposite_bet_pipeline = MultiStudy.translate(
        'dwi_1', 'dwi2ref_opposite_bet_pipeline')

    dwi2ref_1_opposite_motion_alignment_pipeline = MultiStudy.translate(
        'dwi_1', CoregisteredDiffusionStudy.
        dwi2ref_opposite_motion_mat_pipeline)

    dwi2ref_1_opposite_qform_transform_pipeline = MultiStudy.translate(
        'dwi_1',
        CoregisteredDiffusionStudy.dwi2ref_opposite_qform_transform_pipeline)

    dwi2ref_1_opposite_rigid_registration_pipeline = MultiStudy.translate(
        'dwi_1',
        CoregisteredDiffusionStudy.
        dwi2ref_opposite_rigid_registration_pipeline)

    dwi2ref_1_opposite_ref_bet_pipeline = MultiStudy.translate(
        'dwi_1', 'ref_bet_pipeline',
        override_default_options={'bet_method': 'optibet'})

    dwi2ref_1_opposite_ref_basic_preproc_pipeline = MultiStudy.translate(
        'dwi_1', 'ref_basic_preproc_pipeline')

    dwi2ref_1_opposite_main_dcm2nii_pipeline = MultiStudy.translate(
        'dwi_1', CoregisteredDiffusionStudy.
        dwi2ref_opposite_main_dcm2nii_pipeline)

    dwi2ref_1_opposite_ref_dcm2nii_pipeline = MultiStudy.translate(
        'dwi_1', CoregisteredDiffusionStudy.
        dwi2ref_opposite_ref_dcm2nii_pipeline)

    dwi2ref_1_dcm_info_pipeline = MultiStudy.translate(
        'dwi_1', 'dwi2ref_dcm_info_pipeline')

    dwi2ref_1_basic_preproc_pipeline = MultiStudy.translate(
        'dwi_1', 'dwi2ref_topup_pipeline')

    dwi2ref_1_bet_pipeline = MultiStudy.translate(
        'dwi_1', 'dwi2ref_bet_pipeline')

    dwi2ref_1_motion_alignment_pipeline = MultiStudy.translate(
        'dwi_1', 'dwi2ref_motion_mat_pipeline')

    dwi2ref_1_qform_transform_pipeline = MultiStudy.translate(
        'dwi_1',
        CoregisteredDiffusionStudy.dwi2ref_qform_transform_pipeline)

    dwi2ref_1_rigid_registration_pipeline = MultiStudy.translate(
        'dwi_1',
        CoregisteredDiffusionStudy.dwi2ref_rigid_registration_pipeline)

    dwi2ref_1_ref_bet_pipeline = MultiStudy.translate(
        'dwi_1', 'ref_bet_pipeline',
        override_default_options={'bet_method': 'optibet'})

    dwi2ref_1_ref_basic_preproc_pipeline = MultiStudy.translate(
        'dwi_1', 'ref_basic_preproc_pipeline')

    dwi2ref_1_main_dcm2nii_pipeline = MultiStudy.translate(
        'dwi_1', 'dwi2ref_main_dcm2nii_pipeline')

    dwi2ref_1_ref_dcm2nii_pipeline = MultiStudy.translate(
        'dwi_1', 'dwi2ref_ref_dcm2nii_pipeline')

    ref_dcm_info_pipeline = MultiStudy.translate(
        'ref', 'header_info_extraction_pipeline')

    t1_motion_alignment_pipeline = MultiStudy.translate(
        't1_1', 't1_motion_mat_pipeline')

    t1_dcm2nii_pipeline = MultiStudy.translate(
        't1_1', 't1_dcm2nii_pipeline')

    t1_dcm_info_pipeline = MultiStudy.translate(
        't1_1', 't1_dcm_info_pipeline')

    t1_motion_mat_pipeline = MultiStudy.translate(
        't1_1', 't1_motion_mat_pipeline')

    t1_basic_preproc_pipeline = MultiStudy.translate(
        't1_1', 't1_basic_preproc_pipeline')

    t1_qform_transform_pipeline = MultiStudy.translate(
        't1_1', 't1_qform_transform_pipeline')

    t1_bet_pipeline = MultiStudy.translate(
        't1_1', 't1_bet_pipeline',
        override_default_options={'bet_method': 'optibet'})

    t1_ref_bet_pipeline = MultiStudy.translate(
        't1_1', 'ref_bet_pipeline',
        override_default_options={'bet_method': 'optibet'})

    t1_ref_basic_preproc_pipeline = MultiStudy.translate(
        't1_1', 'ref_basic_preproc_pipeline')

    t1_rigid_registration_pipeline = MultiStudy.translate(
        't1_1', 't1_rigid_registration_pipeline')

    ute_motion_alignment_pipeline = MultiStudy.translate(
        'ute', 't1_motion_mat_pipeline')

    ute_dcm2nii_pipeline = MultiStudy.translate(
        'ute', 't1_dcm2nii_pipeline')

    ute_dcm_info_pipeline = MultiStudy.translate(
        'ute', 't1_dcm_info_pipeline')

    ute_motion_mat_pipeline = MultiStudy.translate(
        'ute', 't1_motion_mat_pipeline')

    ute_basic_preproc_pipeline = MultiStudy.translate(
        'ute', 't1_basic_preproc_pipeline')

    ute_qform_transform_pipeline = MultiStudy.translate(
        'ute', 't1_qform_transform_pipeline')

    ute_bet_pipeline = MultiStudy.translate(
        'ute', 't1_bet_pipeline',
        override_default_options={'bet_method': 'optibet'})

    ute_ref_bet_pipeline = MultiStudy.translate(
        'ute', 'ref_bet_pipeline',
        override_default_options={'bet_method': 'optibet'})

    ute_ref_basic_preproc_pipeline = MultiStudy.translate(
        'ute', 'ref_basic_preproc_pipeline')

    ute_rigid_registration_pipeline = MultiStudy.translate(
        'ute', 't1_rigid_registration_pipeline')

    fm_motion_alignment_pipeline = MultiStudy.translate(
        'fm', 't2_motion_mat_pipeline')

    fm_dcm2nii_pipeline = MultiStudy.translate(
        'fm', 't2_dcm2nii_pipeline')

    fm_dcm_info_pipeline = MultiStudy.translate(
        'fm', 't2_dcm_info_pipeline')

    fm_motion_mat_pipeline = MultiStudy.translate(
        'fm', 't2_motion_mat_pipeline')

    fm_basic_preproc_pipeline = MultiStudy.translate(
        'fm', 't2_basic_preproc_pipeline')

    fm_qform_transform_pipeline = MultiStudy.translate(
        'fm', 't2_qform_transform_pipeline')

    fm_bet_pipeline = MultiStudy.translate(
        'fm', 't2_bet_pipeline')

    fm_ref_bet_pipeline = MultiStudy.translate(
        'fm', 'ref_bet_pipeline',
        override_default_options={'bet_method': 'optibet'})

    fm_ref_basic_preproc_pipeline = MultiStudy.translate(
        'fm', 'ref_basic_preproc_pipeline')

    fm_rigid_registration_pipeline = MultiStudy.translate(
        'fm', 't2_rigid_registration_pipeline')

    t2_1_motion_alignment_pipeline = MultiStudy.translate(
        't2_1', 't2_motion_mat_pipeline')

    t2_1_dcm2nii_pipeline = MultiStudy.translate(
        't2_1', 't2_dcm2nii_pipeline')

    t2_1_dcm_info_pipeline = MultiStudy.translate(
        't2_1', 't2_dcm_info_pipeline')

    t2_1_motion_mat_pipeline = MultiStudy.translate(
        't2_1', 't2_motion_mat_pipeline')

    t2_1_basic_preproc_pipeline = MultiStudy.translate(
        't2_1', 't2_basic_preproc_pipeline')

    t2_1_qform_transform_pipeline = MultiStudy.translate(
        't2_1', 't2_qform_transform_pipeline')

    t2_1_bet_pipeline = MultiStudy.translate(
        't2_1', 't2_bet_pipeline')

    t2_1_ref_bet_pipeline = MultiStudy.translate(
        't2_1', 'ref_bet_pipeline',
        override_default_options={'bet_method': 'optibet'})

    t2_1_ref_basic_preproc_pipeline = MultiStudy.translate(
        't2_1', 'ref_basic_preproc_pipeline')

    t2_1_rigid_registration_pipeline = MultiStudy.translate(
        't2_1', 't2_rigid_registration_pipeline')

    t2_2_motion_alignment_pipeline = MultiStudy.translate(
        't2_2', 't2_motion_mat_pipeline')

    t2_2_dcm2nii_pipeline = MultiStudy.translate(
        't2_2', 't2_dcm2nii_pipeline')

    t2_2_dcm_info_pipeline = MultiStudy.translate(
        't2_2', 't2_dcm_info_pipeline')

    t2_2_motion_mat_pipeline = MultiStudy.translate(
        't2_2', 't2_motion_mat_pipeline')

    t2_2_basic_preproc_pipeline = MultiStudy.translate(
        't2_2', 't2_basic_preproc_pipeline')

    t2_2_qform_transform_pipeline = MultiStudy.translate(
        't2_2', 't2_qform_transform_pipeline')

    t2_2_bet_pipeline = MultiStudy.translate(
        't2_2', 't2_bet_pipeline')

    t2_2_ref_bet_pipeline = MultiStudy.translate(
        't2_2', 'ref_bet_pipeline',
        override_default_options={'bet_method': 'optibet'})

    t2_2_ref_basic_preproc_pipeline = MultiStudy.translate(
        't2_2', 'ref_basic_preproc_pipeline')

    t2_2_rigid_registration_pipeline = MultiStudy.translate(
        't2_2', 't2_rigid_registration_pipeline')

    t2_3_motion_alignment_pipeline = MultiStudy.translate(
        't2_3', 't2_motion_mat_pipeline')

    t2_3_dcm2nii_pipeline = MultiStudy.translate(
        't2_3', 't2_dcm2nii_pipeline')

    t2_3_dcm_info_pipeline = MultiStudy.translate(
        't2_3', 't2_dcm_info_pipeline')

    t2_3_motion_mat_pipeline = MultiStudy.translate(
        't2_3', 't2_motion_mat_pipeline')

    t2_3_basic_preproc_pipeline = MultiStudy.translate(
        't2_3', 't2_basic_preproc_pipeline')

    t2_3_qform_transform_pipeline = MultiStudy.translate(
        't2_3', 't2_qform_transform_pipeline')

    t2_3_bet_pipeline = MultiStudy.translate(
        't2_3', 't2_bet_pipeline')

    t2_3_ref_bet_pipeline = MultiStudy.translate(
        't2_3', 'ref_bet_pipeline',
        override_default_options={'bet_method': 'optibet'})

    t2_3_ref_basic_preproc_pipeline = MultiStudy.translate(
        't2_3', 'ref_basic_preproc_pipeline')

    t2_3_rigid_registration_pipeline = MultiStudy.translate(
        't2_3', 't2_rigid_registration_pipeline')

    t2_4_motion_alignment_pipeline = MultiStudy.translate(
        't2_4', 't2_motion_mat_pipeline')

    t2_4_dcm2nii_pipeline = MultiStudy.translate(
        't2_4', 't2_dcm2nii_pipeline')

    t2_4_dcm_info_pipeline = MultiStudy.translate(
        't2_4', 't2_dcm_info_pipeline')

    t2_4_motion_mat_pipeline = MultiStudy.translate(
        't2_4', 't2_motion_mat_pipeline')

    t2_4_basic_preproc_pipeline = MultiStudy.translate(
        't2_4', 't2_basic_preproc_pipeline')

    t2_4_qform_transform_pipeline = MultiStudy.translate(
        't2_4', 't2_qform_transform_pipeline')

    t2_4_bet_pipeline = MultiStudy.translate(
        't2_4', 't2_bet_pipeline')

    t2_4_ref_bet_pipeline = MultiStudy.translate(
        't2_4', 'ref_bet_pipeline',
        override_default_options={'bet_method': 'optibet'})

    t2_4_ref_basic_preproc_pipeline = MultiStudy.translate(
        't2_4', 'ref_basic_preproc_pipeline')

    t2_4_rigid_registration_pipeline = MultiStudy.translate(
        't2_4', 't2_rigid_registration_pipeline')

    t2_5_motion_alignment_pipeline = MultiStudy.translate(
        't2_5', 't2_motion_mat_pipeline')

    t2_5_dcm2nii_pipeline = MultiStudy.translate(
        't2_5', 't2_dcm2nii_pipeline')

    t2_5_dcm_info_pipeline = MultiStudy.translate(
        't2_5', 't2_dcm_info_pipeline')

    t2_5_motion_mat_pipeline = MultiStudy.translate(
        't2_5', 't2_motion_mat_pipeline')

    t2_5_basic_preproc_pipeline = MultiStudy.translate(
        't2_5', 't2_basic_preproc_pipeline')

    t2_5_qform_transform_pipeline = MultiStudy.translate(
        't2_5', 't2_qform_transform_pipeline')

    t2_5_bet_pipeline = MultiStudy.translate(
        't2_5', 't2_bet_pipeline')

    t2_5_ref_bet_pipeline = MultiStudy.translate(
        't2_5', 'ref_bet_pipeline',
        override_default_options={'bet_method': 'optibet'})

    t2_5_ref_basic_preproc_pipeline = MultiStudy.translate(
        't2_5', 'ref_basic_preproc_pipeline')

    t2_5_rigid_registration_pipeline = MultiStudy.translate(
        't2_5', 't2_rigid_registration_pipeline')

    epi1_motion_alignment_pipeline = MultiStudy.translate(
        'epi1', 'epi_motion_alignment_pipeline')

    epi1_dcm2nii_pipeline = MultiStudy.translate(
        'epi1', 'epi_dcm2nii_pipeline')

    epi1_epireg_pipeline = MultiStudy.translate(
        'epi1', 'epireg_pipeline')

    epi1_dcm_info_pipeline = MultiStudy.translate(
        'epi1', 'epi_dcm_info_pipeline')

    epi1_motion_mat_pipeline = MultiStudy.translate(
        'epi1', 'epi_motion_mat_pipeline')

    epi1_basic_preproc_pipeline = MultiStudy.translate(
        'epi1', 'epi_basic_preproc_pipeline')

    epi1_qform_transform_pipeline = MultiStudy.translate(
        'epi1', 'epi_qform_transform_pipeline')

    epi1_bet_pipeline = MultiStudy.translate(
        'epi1', 'epi_bet_pipeline')

    epi1_ref_bet_pipeline = MultiStudy.translate(
        'epi1', 'ref_bet_pipeline',
        override_default_options={'bet_method': 'optibet'})

    epi1_ref_segmentation_pipeline = MultiStudy.translate(
        'epi1', 'ref_segmentation_pipeline',
        override_default_options={'img_type': 1})

    epi1_ref_basic_preproc_pipeline = MultiStudy.translate(
        'epi1', 'ref_basic_preproc_pipeline')

    epi1_ref_nifti_pipeline = MultiStudy.translate(
        'epi1', 'ref_dcm2nii_pipeline')

    def mean_displacement_pipeline(self, **kwargs):

        pipeline = self.create_pipeline(
            name='mean_displacement_calculation',
            inputs=[DatasetSpec('t1_1_motion_mats', directory_format),
                    DatasetSpec('ref_motion_mats', directory_format),
                    DatasetSpec('ute_motion_mats', directory_format),
                    DatasetSpec('fm_motion_mats', directory_format),
                    DatasetSpec('epi1_motion_mats', directory_format),
                    DatasetSpec('t2_1_motion_mats', directory_format),
                    DatasetSpec('t2_2_motion_mats', directory_format),
                    DatasetSpec('t2_3_motion_mats', directory_format),
                    DatasetSpec('t2_4_motion_mats', directory_format),
                    DatasetSpec('t2_5_motion_mats', directory_format),
                    DatasetSpec('dwi_1_main_motion_mats', directory_format),
                    DatasetSpec('dwi2ref_1_motion_mats', directory_format),
                    DatasetSpec('dwi2ref_1_opposite_motion_mats',
                                directory_format),
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
                    FieldSpec('t2_1_tr', float),
                    FieldSpec('t2_1_start_time', str),
                    FieldSpec('t2_1_real_duration', str),
                    FieldSpec('t2_2_tr', float),
                    FieldSpec('t2_2_start_time', str),
                    FieldSpec('t2_2_real_duration', str),
                    FieldSpec('t2_3_tr', float),
                    FieldSpec('t2_3_start_time', str),
                    FieldSpec('t2_3_real_duration', str),
                    FieldSpec('t2_4_tr', float),
                    FieldSpec('t2_4_start_time', str),
                    FieldSpec('t2_4_real_duration', str),
                    FieldSpec('t2_5_tr', float),
                    FieldSpec('t2_5_start_time', str),
                    FieldSpec('t2_5_real_duration', str),
                    FieldSpec('dwi_1_main_tr', float),
                    FieldSpec('dwi_1_main_start_time', str),
                    FieldSpec('dwi_1_main_real_duration', str),
                    FieldSpec('dwi2ref_1_tr', float),
                    FieldSpec('dwi2ref_1_start_time', str),
                    FieldSpec('dwi2ref_1_real_duration', str),
                    FieldSpec('dwi2ref_1_opposite_tr', float),
                    FieldSpec('dwi2ref_1_opposite_start_time', str),
                    FieldSpec('dwi2ref_1_opposite_real_duration', str)],
            outputs=[DatasetSpec('mean_displacement', text_format),
                     DatasetSpec('mean_displacement_rc', text_format),
                     DatasetSpec('mean_displacement_consecutive', text_format),
                     DatasetSpec('start_times', text_format),
                     DatasetSpec('motion_par_rc', text_format),
                     DatasetSpec('offset_indexes', text_format),
                     DatasetSpec('mats4average', text_format)],
            description=("Calculate the mean displacement between each motion"
                         " matrix and a reference."),
            version=1,
            citations=[fsl_cite],
            **kwargs)

        merge_ref = pipeline.create_node(merge_lists(4), name='merge_ref')
        pipeline.connect_input('ref_motion_mats', merge_ref, 'in1')
        pipeline.connect_input('ref_start_time', merge_ref, 'in2')
        pipeline.connect_input('ref_real_duration', merge_ref, 'in3')
        pipeline.connect_input('ref_tr', merge_ref, 'in4')

        merge_t1_1 = pipeline.create_node(merge_lists(4), name='merge_t1_1')
        pipeline.connect_input('t1_1_motion_mats', merge_t1_1, 'in1')
        pipeline.connect_input('t1_1_start_time', merge_t1_1, 'in2')
        pipeline.connect_input('t1_1_real_duration', merge_t1_1, 'in3')
        pipeline.connect_input('t1_1_tr', merge_t1_1, 'in4')

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

        merge_t2_1 = pipeline.create_node(merge_lists(4), name='merge_t2_1')
        pipeline.connect_input('t2_1_motion_mats', merge_t2_1, 'in1')
        pipeline.connect_input('t2_1_start_time', merge_t2_1, 'in2')
        pipeline.connect_input('t2_1_real_duration', merge_t2_1, 'in3')
        pipeline.connect_input('t2_1_tr', merge_t2_1, 'in4')

        merge_t2_2 = pipeline.create_node(merge_lists(4), name='merge_t2_2')
        pipeline.connect_input('t2_2_motion_mats', merge_t2_2, 'in1')
        pipeline.connect_input('t2_2_start_time', merge_t2_2, 'in2')
        pipeline.connect_input('t2_2_real_duration', merge_t2_2, 'in3')
        pipeline.connect_input('t2_2_tr', merge_t2_2, 'in4')

        merge_t2_3 = pipeline.create_node(merge_lists(4), name='merge_t2_3')
        pipeline.connect_input('t2_3_motion_mats', merge_t2_3, 'in1')
        pipeline.connect_input('t2_3_start_time', merge_t2_3, 'in2')
        pipeline.connect_input('t2_3_real_duration', merge_t2_3, 'in3')
        pipeline.connect_input('t2_3_tr', merge_t2_3, 'in4')

        merge_t2_4 = pipeline.create_node(merge_lists(4), name='merge_t2_4')
        pipeline.connect_input('t2_4_motion_mats', merge_t2_4, 'in1')
        pipeline.connect_input('t2_4_start_time', merge_t2_4, 'in2')
        pipeline.connect_input('t2_4_real_duration', merge_t2_4, 'in3')
        pipeline.connect_input('t2_4_tr', merge_t2_4, 'in4')

        merge_t2_5 = pipeline.create_node(merge_lists(4), name='merge_t2_5')
        pipeline.connect_input('t2_5_motion_mats', merge_t2_5, 'in1')
        pipeline.connect_input('t2_5_start_time', merge_t2_5, 'in2')
        pipeline.connect_input('t2_5_real_duration', merge_t2_5, 'in3')
        pipeline.connect_input('t2_5_tr', merge_t2_5, 'in4')

        merge_epi1 = pipeline.create_node(merge_lists(4), name='merge_epi1')
        pipeline.connect_input('epi1_motion_mats', merge_epi1, 'in1')
        pipeline.connect_input('epi1_start_time', merge_epi1, 'in2')
        pipeline.connect_input('epi1_real_duration', merge_epi1, 'in3')
        pipeline.connect_input('epi1_tr', merge_epi1, 'in4')

        merge_dwi_1_main = pipeline.create_node(merge_lists(4),
                                                name='merge_dwi_1_main')
        pipeline.connect_input('dwi_1_main_motion_mats', merge_dwi_1_main,
                               'in1')
        pipeline.connect_input('dwi_1_main_start_time', merge_dwi_1_main,
                               'in2')
        pipeline.connect_input('dwi_1_main_real_duration', merge_dwi_1_main,
                               'in3')
        pipeline.connect_input('dwi_1_main_tr', merge_dwi_1_main, 'in4')

        merge_dwi2ref_1 = pipeline.create_node(
            merge_lists(4), name='merge_dwi2ref_1')
        pipeline.connect_input(
            'dwi2ref_1_motion_mats', merge_dwi2ref_1, 'in1')
        pipeline.connect_input(
            'dwi2ref_1_start_time', merge_dwi2ref_1, 'in2')
        pipeline.connect_input(
            'dwi2ref_1_real_duration', merge_dwi2ref_1, 'in3')
        pipeline.connect_input(
            'dwi2ref_1_tr', merge_dwi2ref_1, 'in4')

        merge_dwi2ref_1_opposite = pipeline.create_node(
            merge_lists(4), name='merge_dwi2ref_1_opposite')
        pipeline.connect_input(
            'dwi2ref_1_opposite_motion_mats', merge_dwi2ref_1_opposite, 'in1')
        pipeline.connect_input(
            'dwi2ref_1_opposite_start_time', merge_dwi2ref_1_opposite, 'in2')
        pipeline.connect_input(
            'dwi2ref_1_opposite_real_duration', merge_dwi2ref_1_opposite,
            'in3')
        pipeline.connect_input(
            'dwi2ref_1_opposite_tr', merge_dwi2ref_1_opposite, 'in4')

        merge_scans = pipeline.create_node(merge_lists(13), name='merge_scans')
        merge_scans.inputs.no_flatten = True
        pipeline.connect(merge_epi1, 'out', merge_scans, 'in1')
        pipeline.connect(merge_t2_1, 'out', merge_scans, 'in2')
        pipeline.connect(merge_t2_2, 'out', merge_scans, 'in3')
        pipeline.connect(merge_t2_3, 'out', merge_scans, 'in4')
        pipeline.connect(merge_t2_4, 'out', merge_scans, 'in5')
        pipeline.connect(merge_t1_1, 'out', merge_scans, 'in6')
        pipeline.connect(merge_ute, 'out', merge_scans, 'in7')
        pipeline.connect(merge_fm, 'out', merge_scans, 'in8')
        pipeline.connect(merge_ref, 'out', merge_scans, 'in9')
        pipeline.connect(merge_dwi_1_main, 'out', merge_scans, 'in10')
        pipeline.connect(merge_t2_5, 'out', merge_scans, 'in11')
        pipeline.connect(merge_dwi2ref_1_opposite, 'out', merge_scans, 'in12')
        pipeline.connect(merge_dwi2ref_1, 'out', merge_scans, 'in13')

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
        pipeline.connect_output('mats4average', md, 'mats4average')
        pipeline.assert_connected()
        return pipeline

    def motion_framing_pipeline(self, **kwargs):

        pipeline = self.create_pipeline(
            name='motion_framing',
            inputs=[DatasetSpec('mean_displacement', text_format),
                    DatasetSpec('mean_displacement_consecutive', text_format),
                    DatasetSpec('start_times', text_format)],
            outputs=[DatasetSpec('frame_start_times', text_format),
                     DatasetSpec('frame_vol_numbers', text_format)],
            description=("Calculate when the head movement exceeded a "
                         "predefined threshold (default 2mm)."),
            default_options={'th': 2.0, 'temporal_th': 30.0},
            version=1,
            citations=[fsl_cite],
            **kwargs)

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
        pipeline.connect_output('frame_vol_numbers', framing,
                                'frame_vol_numbers')
        pipeline.assert_connected()
        return pipeline

    def plot_mean_displacement_pipeline(self, **kwargs):

        pipeline = self.create_pipeline(
            name='plot_mean_displacement',
            inputs=[DatasetSpec('mean_displacement_rc', text_format),
                    DatasetSpec('offset_indexes', text_format),
                    DatasetSpec('frame_start_times', text_format)],
            outputs=[DatasetSpec('mean_displacement_plot', png_format)],
            description=("Plot the mean displacement real clock"),
            default_options={'framing': True},
            version=1,
            citations=[fsl_cite],
            **kwargs)

        plot_md = pipeline.create_node(PlotMeanDisplacementRC(),
                                       name='plot_md')
        plot_md.inputs.framing = pipeline.option('framing')
        pipeline.connect_input('mean_displacement_rc', plot_md,
                               'mean_disp_rc')
        pipeline.connect_input('offset_indexes', plot_md,
                               'false_indexes')
        pipeline.connect_input('frame_start_times', plot_md,
                               'frame_start_times')
        pipeline.connect_output('mean_displacement_plot', plot_md,
                                'mean_disp_plot')
        pipeline.assert_connected()
        return pipeline

    def frame_mean_transformation_mats_pipeline(self, **kwargs):

        pipeline = self.create_pipeline(
            name='frame_mean_transformation_mats',
            inputs=[DatasetSpec('mats4average', text_format),
                    DatasetSpec('frame_vol_numbers', text_format)],
            outputs=[DatasetSpec('average_mats', directory_format)],
            description=("Average all the transformation mats within each "
                         "detected frame."),
            version=1,
            citations=[fsl_cite],
            **kwargs)

        average = pipeline.create_node(AffineMatAveraging(),
                                       name='mats_averaging')
        pipeline.connect_input('frame_vol_numbers', average,
                               'frame_vol_numbers')
        pipeline.connect_input('mats4average', average,
                               'all_mats4average')
        pipeline.connect_output('average_mats', average,
                                'average_mats')
        pipeline.assert_connected()
        return pipeline

    def pet_correction_factors_pipeline(self, **kwargs):

        pipeline = self.create_pipeline(
            name='pet_correction_factors',
            inputs=[DatasetSpec('frame_start_times', text_format)],
            outputs=[DatasetSpec('correction_factors', text_format)],
            description=("Pipeline to calculate the correction factors to "
                         "account for frame duration when averaging the PET "
                         "frames to create the static PET image"),
            version=1,
            citations=[fsl_cite],
            **kwargs)

        corr_factors = pipeline.create_node(PetCorrectionFactor(),
                                            name='pet_corr_factors')
        pipeline.connect_input('frame_start_times', corr_factors,
                               'frame_start_times')
        pipeline.connect_output('correction_factors', corr_factors,
                                'corr_factors')
        pipeline.assert_connected()
        return pipeline

    def frame2ref_alignment_pipeline_factory(
            self, name, average_mats, ute_regmat, ute_qform_mat, umap=None,
            pct=False, fixed_binning=False, **options):
        inputs = [DatasetSpec(average_mats, directory_format),
                  DatasetSpec(ute_regmat, text_matrix_format),
                  DatasetSpec(ute_qform_mat, text_matrix_format)]
        outputs = [DatasetSpec('frame2reference_mats', directory_format)]
        if umap:
            inputs.append(DatasetSpec(umap, nifti_gz_format))
            outputs.append(DatasetSpec('umaps_align2ref', directory_format))

        pipeline = self.create_pipeline(
            name=name,
            inputs=inputs,
            outputs=outputs,
            description=("Pipeline to create an affine mat to align each "
                         "detected frame to the reference. If umap is provided"
                         ", it will be also aligned to match the head position"
                         " in each frame and improve the static PET image "
                         "quality."),
            default_options={'pct': pct, 'fixed_binning': fixed_binning},
            version=1,
            citations=[fsl_cite],
            **kwargs)

        frame_align = pipeline.create_node(
            FrameAlign2Reference(), name='frame2ref_alignment',
            requirements=[fsl509_req])
        frame_align.inputs.pct = pipeline.option('pct')
        frame_align.inputs.fixed_binning = pipeline.option('fixed_binning')
        pipeline.connect_input(average_mats, frame_align,
                               'average_mats')
        pipeline.connect_input(ute_regmat, frame_align,
                               'ute_regmat')
        pipeline.connect_input(ute_qform_mat, frame_align,
                               'ute_qform_mat')
        if umap:
            pipeline.connect_input(umap, frame_align, 'umap')
            pipeline.connect_output('umaps_align2ref', frame_align,
                                    'umaps_align2ref')
        pipeline.connect_output('frame2reference_mats', frame_align,
                                'frame2reference_mats')
        pipeline.assert_connected()
        return pipeline

    def frame2ref_alignment_pipeline(self, **kwargs):
        return self.frame2ref_alignment_pipeline_factory(
            'frame2ref_alignment', 'average_mats', 'ute_reg_mat',
            'ute_qform_mat', umap='umap',
            pct=False, fixed_binning=False, **options)

    sub_study_specs = [
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
        SubStudySpec('t2_1', CoregisteredT2Study, {
            't2_1': 't2',
            't2_1_nifti': 't2_nifti',
            't2_1_reg': 't2_reg',
            't2_1_reg_mat': 't2_reg_mat',
            't2_1_qformed': 't2_qformed',
            't2_1_qform_mat': 't2_qform_mat',
            't2_1_reference': 'reference',
            't2_1_brain': 't2_brain',
            't2_1_brain_mask': 't2_brain_mask',
            't2_1_preproc': 't2_preproc',
            't2_1_ref_preproc': 'ref_preproc',
            't2_1_ref_brain': 'ref_brain',
            't2_1_ref_brain_mask': 'ref_brain_mask',
            't2_1_ped': 't2_ped',
            't2_1_pe_angle': 't2_pe_angle',
            't2_1_tr': 't2_tr',
            't2_1_real_duration': 't2_real_duration',
            't2_1_tot_duration': 't2_tot_duration',
            't2_1_start_time': 't2_start_time',
            't2_1_dcm_info': 't2_dcm_info',
            't2_1_motion_mats': 't2_motion_mats'}),
        SubStudySpec('t2_2', CoregisteredT2Study, {
            't2_2': 't2',
            't2_2_nifti': 't2_nifti',
            't2_2_reg': 't2_reg',
            't2_2_reg_mat': 't2_reg_mat',
            't2_2_qformed': 't2_qformed',
            't2_2_qform_mat': 't2_qform_mat',
            't2_2_reference': 'reference',
            't2_2_brain': 't2_brain',
            't2_2_brain_mask': 't2_brain_mask',
            't2_2_preproc': 't2_preproc',
            't2_2_ref_preproc': 'ref_preproc',
            't2_2_ref_brain': 'ref_brain',
            't2_2_ref_brain_mask': 'ref_brain_mask',
            't2_2_ped': 't2_ped',
            't2_2_pe_angle': 't2_pe_angle',
            't2_2_tr': 't2_tr',
            't2_2_real_duration': 't2_real_duration',
            't2_2_tot_duration': 't2_tot_duration',
            't2_2_start_time': 't2_start_time',
            't2_2_dcm_info': 't2_dcm_info',
            't2_2_motion_mats': 't2_motion_mats'}),
        SubStudySpec('t2_3', CoregisteredT2Study, {
            't2_3': 't2',
            't2_3_nifti': 't2_nifti',
            't2_3_reg': 't2_reg',
            't2_3_reg_mat': 't2_reg_mat',
            't2_3_qformed': 't2_qformed',
            't2_3_qform_mat': 't2_qform_mat',
            't2_3_reference': 'reference',
            't2_3_brain': 't2_brain',
            't2_3_brain_mask': 't2_brain_mask',
            't2_3_preproc': 't2_preproc',
            't2_3_ref_preproc': 'ref_preproc',
            't2_3_ref_brain': 'ref_brain',
            't2_3_ref_brain_mask': 'ref_brain_mask',
            't2_3_ped': 't2_ped',
            't2_3_pe_angle': 't2_pe_angle',
            't2_3_tr': 't2_tr',
            't2_3_real_duration': 't2_real_duration',
            't2_3_tot_duration': 't2_tot_duration',
            't2_3_start_time': 't2_start_time',
            't2_3_dcm_info': 't2_dcm_info',
            't2_3_motion_mats': 't2_motion_mats'}),
        SubStudySpec('t2_4', CoregisteredT2Study, {
            't2_4': 't2',
            't2_4_nifti': 't2_nifti',
            't2_4_reg': 't2_reg',
            't2_4_reg_mat': 't2_reg_mat',
            't2_4_qformed': 't2_qformed',
            't2_4_qform_mat': 't2_qform_mat',
            't2_4_reference': 'reference',
            't2_4_brain': 't2_brain',
            't2_4_brain_mask': 't2_brain_mask',
            't2_4_preproc': 't2_preproc',
            't2_4_ref_preproc': 'ref_preproc',
            't2_4_ref_brain': 'ref_brain',
            't2_4_ref_brain_mask': 'ref_brain_mask',
            't2_4_ped': 't2_ped',
            't2_4_pe_angle': 't2_pe_angle',
            't2_4_tr': 't2_tr',
            't2_4_real_duration': 't2_real_duration',
            't2_4_tot_duration': 't2_tot_duration',
            't2_4_start_time': 't2_start_time',
            't2_4_dcm_info': 't2_dcm_info',
            't2_4_motion_mats': 't2_motion_mats'}),
        SubStudySpec('t2_5', CoregisteredT2Study, {
            't2_5': 't2',
            't2_5_nifti': 't2_nifti',
            't2_5_reg': 't2_reg',
            't2_5_reg_mat': 't2_reg_mat',
            't2_5_qformed': 't2_qformed',
            't2_5_qform_mat': 't2_qform_mat',
            't2_5_reference': 'reference',
            't2_5_brain': 't2_brain',
            't2_5_brain_mask': 't2_brain_mask',
            't2_5_preproc': 't2_preproc',
            't2_5_ref_preproc': 'ref_preproc',
            't2_5_ref_brain': 'ref_brain',
            't2_5_ref_brain_mask': 'ref_brain_mask',
            't2_5_ped': 't2_ped',
            't2_5_pe_angle': 't2_pe_angle',
            't2_5_tr': 't2_tr',
            't2_5_real_duration': 't2_real_duration',
            't2_5_tot_duration': 't2_tot_duration',
            't2_5_start_time': 't2_start_time',
            't2_5_dcm_info': 't2_dcm_info',
            't2_5_motion_mats': 't2_motion_mats'}),
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
            'asl_dcm_info': 'epi_dcm_info'}),
        SubStudySpec('dwi_1', CoregisteredDiffusionStudy, {
            'dwi_1_main': 'dwi_main',
            'dwi_1_main_ref': 'dwi_main_ref',
            'dwi_1_main_brain_mask': 'dwi_main_brain_mask',
            'dwi_1_main_brain': 'dwi_main_brain',
            'dwi_1_main_preproc': 'dwi_main_preproc',
            'dwi_1_main_eddy_par': 'dwi_main_eddy_par',
            'dwi_1_main_ped': 'dwi_main_ped',
            'dwi_1_main_pe_angle': 'dwi_main_pe_angle',
            'dwi_1_main_tr': 'dwi_main_tr',
            'dwi_1_main_real_duration': 'dwi_main_real_duration',
            'dwi_1_main_tot_duration': 'dwi_main_tot_duration',
            'dwi_1_main_start_time': 'dwi_main_start_time',
            'dwi_1_main_dcm_info': 'dwi_main_dcm_info',
            'dwi_1_main_affine_mats': 'dwi_main_affine_mats',
            'dwi_1_main_motion_mats': 'dwi_main_motion_mats',
            'dwi_1_main_reg_mat': 'dwi_main_reg_mat',
            'dwi_1_main_reg': 'dwi_main_reg',
            'dwi_1_main_qform_mat': 'dwi_main_qform_mat',
            'dwi_1_main_qformed': 'dwi_main_qformed',
            'dwi2ref_1_opposite_to_correct': 'dwi2ref_opposite_to_correct',
            'dwi2ref_1_opposite_ref': 'dwi2ref_opposite_ref',
            'dwi2ref_1_opposite_ref_nii': 'dwi2ref_opposite_ref_nii',
            'dwi2ref_1_opposite_to_correct_nii': ('dwi2ref_opposite_to_correct'
                                                  '_nii'),
            'dwi2ref_1_opposite_brain_mask': 'dwi2ref_opposite_brain_mask',
            'dwi2ref_1_opposite_brain': 'dwi2ref_opposite_brain',
            'dwi2ref_1_opposite_preproc': 'dwi2ref_opposite_preproc',
            'dwi2ref_1_opposite_ped': 'dwi2ref_opposite_ped',
            'dwi2ref_1_opposite_start_time': 'dwi2ref_opposite_start_time',
            'dwi2ref_1_opposite_pe_angle': 'dwi2ref_opposite_pe_angle',
            'dwi2ref_1_opposite_tr': 'dwi2ref_opposite_tr',
            'dwi2ref_1_opposite_dcm_info': 'dwi2ref_opposite_dcm_info',
            'dwi2ref_1_opposite_real_duration': ('dwi2ref_opposite_real_'
                                                 'duration'),
            'dwi2ref_1_opposite_tot_duration': 'dwi2ref_opposite_tot_duration',
            'dwi2ref_1_opposite_start_time': 'dwi2ref_opposite_start_time',
            'dwi2ref_1_opposite_motion_mats': 'dwi2ref_opposite_motion_mats',
            'dwi2ref_1_opposite_reg_mat': 'dwi2ref_opposite_reg_mat',
            'dwi2ref_1_opposite_reg': 'dwi2ref_opposite_reg',
            'dwi2ref_1_opposite_qform_mat': 'dwi2ref_opposite_qform_mat',
            'dwi2ref_1_opposite_qformed': 'dwi2ref_opposite_qformed',
            'dwi2ref_1_to_correct': 'dwi2ref_to_correct',
            'dwi2ref_1_ref': 'dwi2ref_ref',
            'dwi2ref_1_ref_nii': 'dwi2ref_ref_nii',
            'dwi2ref_1_to_correct_nii': 'dwi2ref_to_correct_nii',
            'dwi2ref_1_brain_mask': 'dwi2ref_brain_mask',
            'dwi2ref_1_brain': 'dwi2ref_brain',
            'dwi2ref_1_preproc': 'dwi2ref_preproc',
            'dwi2ref_1_ped': 'dwi2ref_ped',
            'dwi2ref_1_start_time': 'dwi2ref_start_time',
            'dwi2ref_1_pe_angle': 'dwi2ref_pe_angle',
            'dwi2ref_1_tr': 'dwi2ref_tr',
            'dwi2ref_1_dcm_info': 'dwi2ref_dcm_info',
            'dwi2ref_1_real_duration': 'dwi2ref_real_duration',
            'dwi2ref_1_tot_duration': 'dwi2ref_tot_duration',
            'dwi2ref_1_start_time': 'dwi2ref_start_time',
            'dwi2ref_1_motion_mats': 'dwi2ref_motion_mats',
            'dwi2ref_1_reg_mat': 'dwi2ref_reg_mat',
            'dwi2ref_1_reg': 'dwi2ref_reg',
            'dwi2ref_1_qform_mat': 'dwi2ref_qform_mat',
            'dwi2ref_1_qformed': 'dwi2ref_qformed',
            'dwi_ref_preproc': 'ref_preproc',
            'dwi_ref_brain': 'ref_brain',
            'dwi_ref_brain_mask': 'ref_brain_mask',
            'dwi_reference': 'reference'})]

    add_data_specs = [
        DatasetSpec('dwi_1_main', dicom_format),
        DatasetSpec('dwi_1_main_ref', nifti_gz_format),
        DatasetSpec('umap', nifti_gz_format),
        DatasetSpec('dwi_reference', nifti_gz_format),
        DatasetSpec('dwi_1_main_brain_mask', nifti_gz_format,
                    'dwi_1_main_bet_pipeline'),
        DatasetSpec('dwi_1_main_brain', nifti_gz_format,
                    'dwi_1_main_bet_pipeline'),
        DatasetSpec('dwi_1_main_preproc', nifti_gz_format,
                    'dwi_1_main_basic_preproc_pipeline'),
        DatasetSpec('dwi_1_main_eddy_par', eddy_par_format,
                    'dwi_1_main_basic_preproc_pipeline'),
        DatasetSpec('dwi_1_main_dcm_info', text_format,
                    'dwi_1_main_dcm_info_pipeline'),
        FieldSpec('dwi_1_main_ped', str, 'dwi_1_main_dcm_info_pipeline'),
        FieldSpec('dwi_1_main_pe_angle', str, 'dwi_1_main_dcm_info_pipeline'),
        FieldSpec('dwi_1_main_tr', float, 'dwi_1_main_dcm_info_pipeline'),
        FieldSpec('dwi_1_main_real_duration', str,
                  'dwi_1_main_dcm_info_pipeline'),
        FieldSpec('dwi_1_main_tot_duration', str,
                  'dwi_1_main_dcm_info_pipeline'),
        FieldSpec('dwi_1_main_start_time', str, 'dwi_1_main_dcm_info_pipeline'),
        DatasetSpec('dwi_1_main_affine_mats', directory_format,
                    'dwi_1_main_aff_mat_pipeline'),
        DatasetSpec('dwi_1_main_motion_mats', directory_format,
                    'dwi_1_main_motion_alignment_pipeline'),
        DatasetSpec('dwi_1_main_reg_mat', text_matrix_format,
                    'dwi_1_main_rigid_registration_pipeline'),
        DatasetSpec('dwi_1_main_reg', nifti_gz_format,
                    'dwi_1_main_rigid_registration_pipeline'),
        DatasetSpec('dwi_1_main_qform_mat', text_matrix_format,
                    'dwi_1_main_qform_transform_pipeline'),
        DatasetSpec('dwi_1_main_qformed', nifti_gz_format,
                    'dwi_1_main_qform_transform_pipeline'),
        DatasetSpec('dwi_ref_preproc', nifti_gz_format,
                    'dwi_1_main_ref_basic_preproc_pipeline'),
        DatasetSpec('dwi_ref_brain', nifti_gz_format,
                    'dwi_1_main_ref_bet_pipeline'),
        DatasetSpec('dwi_ref_brain_mask', nifti_gz_format,
                    'dwi_1_main_ref_bet_pipeline'),
        DatasetSpec('dwi2ref_1_opposite_to_correct', dicom_format),
        DatasetSpec('dwi2ref_1_opposite_ref', dicom_format),
        DatasetSpec('dwi2ref_1_opposite_to_correct_nii', nifti_gz_format,
                    'dwi2ref_1_opposite_main_dcm2nii_pipeline'),
        DatasetSpec('dwi2ref_1_opposite_ref_nii', nifti_gz_format,
                    'dwi2ref_1_opposite_ref_dcm2nii_pipeline'),
        DatasetSpec('dwi2ref_1_opposite_brain_mask', nifti_gz_format,
                    'dwi2ref_1_opposite_bet_pipeline'),
        DatasetSpec('dwi2ref_1_opposite_brain', nifti_gz_format,
                    'dwi2ref_1_opposite_bet_pipeline'),
        DatasetSpec('dwi2ref_1_opposite_preproc', nifti_gz_format,
                    'dwi2ref_1_opposite_basic_preproc_pipeline'),
        DatasetSpec('dwi2ref_1_opposite_dcm_info', text_format,
                    'dwi2ref_1_opposite_dcm_info_pipeline'),
        FieldSpec('dwi2ref_1_opposite_ped', str,
                  'dwi2ref_1_opposite_dcm_info_pipeline'),
        FieldSpec('dwi2ref_1_opposite_pe_angle', str,
                  'dwi2ref_1_opposite_dcm_info_pipeline'),
        FieldSpec('dwi2ref_1_opposite_tr', float,
                  'dwi2ref_1_opposite_dcm_info_pipeline'),
        FieldSpec('dwi2ref_1_opposite_real_duration', str,
                  'dwi2ref_1_opposite_dcm_info_pipeline'),
        FieldSpec('dwi2ref_1_opposite_tot_duration', str,
                  'dwi2ref_1_opposite_dcm_info_pipeline'),
        FieldSpec('dwi2ref_1_opposite_start_time', str,
                  'dwi2ref_1_opposite_dcm_info_pipeline'),
        DatasetSpec('dwi2ref_1_opposite_motion_mats', directory_format,
                    'dwi2ref_1_opposite_motion_alignment_pipeline'),
        DatasetSpec('dwi2ref_1_opposite_reg_mat', text_matrix_format,
                    'dwi2ref_1_opposite_rigid_registration_pipeline'),
        DatasetSpec('dwi2ref_1_opposite_reg', nifti_gz_format,
                    'dwi2ref_1_opposite_rigid_registration_pipeline'),
        DatasetSpec('dwi2ref_1_opposite_qform_mat', text_matrix_format,
                    'dwi2ref_1_opposite_qform_transform_pipeline'),
        DatasetSpec('dwi2ref_1_opposite_qformed', nifti_gz_format,
                    'dwi2ref_1_opposite_qform_transform_pipeline'),
        DatasetSpec('dwi2ref_1_to_correct', dicom_format),
        DatasetSpec('dwi2ref_1_ref', dicom_format),
        DatasetSpec('dwi2ref_1_to_correct_nii', nifti_gz_format,
                    'dwi2ref_1_main_dcm2nii_pipeline'),
        DatasetSpec('dwi2ref_1_ref_nii', nifti_gz_format,
                    'dwi2ref_1_ref_dcm2nii_pipeline'),
        DatasetSpec('dwi2ref_1_brain_mask', nifti_gz_format,
                    'dwi2ref_1_bet_pipeline'),
        DatasetSpec('dwi2ref_1_brain', nifti_gz_format,
                    'dwi2ref_1_bet_pipeline'),
        DatasetSpec('dwi2ref_1_preproc', nifti_gz_format,
                    'dwi2ref_1_basic_preproc_pipeline'),
        DatasetSpec('dwi2ref_1_dcm_info', text_format,
                    'dwi2ref_1_dcm_info_pipeline'),
        FieldSpec('dwi2ref_1_ped', str, 'dwi2ref_1_dcm_info_pipeline'),
        FieldSpec('dwi2ref_1_pe_angle', str,
                  'dwi2ref_1_dcm_info_pipeline'),
        FieldSpec('dwi2ref_1_tr', float,
                  'dwi2ref_1_dcm_info_pipeline'),
        FieldSpec('dwi2ref_1_real_duration', str,
                  'dwi2ref_1_dcm_info_pipeline'),
        FieldSpec('dwi2ref_1_tot_duration', str,
                  'dwi2ref_1_dcm_info_pipeline'),
        FieldSpec('dwi2ref_1_start_time', str,
                  'dwi2ref_1_dcm_info_pipeline'),
        DatasetSpec('dwi2ref_1_motion_mats', directory_format,
                    'dwi2ref_1_motion_alignment_pipeline'),
        DatasetSpec('dwi2ref_1_reg_mat', text_matrix_format,
                    'dwi2ref_1_rigid_registration_pipeline'),
        DatasetSpec('dwi2ref_1_reg', nifti_gz_format,
                    'dwi2ref_1_rigid_registration_pipeline'),
        DatasetSpec('dwi2ref_1_qform_mat', text_matrix_format,
                    'dwi2ref_1_qform_transform_pipeline'),
        DatasetSpec('dwi2ref_1_qformed', nifti_gz_format,
                    'dwi2ref_1_qform_transform_pipeline'),
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
        DatasetSpec('t2_1', dicom_format),
        DatasetSpec('t2_1_reference', nifti_gz_format),
        DatasetSpec('t2_1_nifti', nifti_gz_format, 't2_1_dcm2nii_pipeline'),
        DatasetSpec('t2_1_reg', nifti_gz_format,
                    't2_1_rigid_registration_pipeline'),
        DatasetSpec('t2_1_reg_mat', text_matrix_format,
                    't2_1_rigid_registration_pipeline'),
        DatasetSpec('t2_1_qformed', nifti_gz_format,
                    't2_1_qform_transform_pipeline'),
        DatasetSpec('t2_1_qform_mat', text_matrix_format,
                    't2_1_qform_transform_pipeline'),
        DatasetSpec('t2_1_brain', nifti_gz_format, 't2_1_bet_pipeline'),
        DatasetSpec('t2_1_brain_mask', nifti_gz_format, 't2_1_bet_pipeline'),
        DatasetSpec('t2_1_preproc', nifti_gz_format,
                    't2_1_basic_preproc_pipeline'),
        DatasetSpec('t2_1_ref_preproc', nifti_gz_format,
                    't2_1_ref_basic_preproc_pipeline'),
        DatasetSpec('t2_1_ref_brain', nifti_gz_format, 't2_1_ref_bet_pipeline'),
        DatasetSpec('t2_1_ref_brain_mask', nifti_gz_format,
                    't2_1_ref_bet_pipeline'),
        DatasetSpec('t2_1_motion_mats', directory_format,
                    't2_1_motion_mat_pipeline'),
        DatasetSpec('t2_1_dcm_info', text_format,
                    't2_1_dcm_info_pipeline'),
        FieldSpec('t2_1_ped', str, 't2_1_dcm_info_pipeline'),
        FieldSpec('t2_1_pe_angle', str, 't2_1_dcm_info_pipeline'),
        FieldSpec('t2_1_tr', float, 't2_1_dcm_info_pipeline'),
        FieldSpec('t2_1_start_time', str, 't2_1_dcm_info_pipeline'),
        FieldSpec('t2_1_real_duration', str, 't2_1_dcm_info_pipeline'),
        FieldSpec('t2_1_tot_duration', str, 't2_1_dcm_info_pipeline'),
        DatasetSpec('t2_2', dicom_format),
        DatasetSpec('t2_2_reference', nifti_gz_format),
        DatasetSpec('t2_2_nifti', nifti_gz_format, 't2_2_dcm2nii_pipeline'),
        DatasetSpec('t2_2_reg', nifti_gz_format,
                    't2_2_rigid_registration_pipeline'),
        DatasetSpec('t2_2_reg_mat', text_matrix_format,
                    't2_2_rigid_registration_pipeline'),
        DatasetSpec('t2_2_qformed', nifti_gz_format,
                    't2_2_qform_transform_pipeline'),
        DatasetSpec('t2_2_qform_mat', text_matrix_format,
                    't2_2_qform_transform_pipeline'),
        DatasetSpec('t2_2_brain', nifti_gz_format, 't2_2_bet_pipeline'),
        DatasetSpec('t2_2_brain_mask', nifti_gz_format, 't2_2_bet_pipeline'),
        DatasetSpec('t2_2_preproc', nifti_gz_format,
                    't2_2_basic_preproc_pipeline'),
        DatasetSpec('t2_2_ref_preproc', nifti_gz_format,
                    't2_2_ref_basic_preproc_pipeline'),
        DatasetSpec('t2_2_ref_brain', nifti_gz_format, 't2_2_ref_bet_pipeline'),
        DatasetSpec('t2_2_ref_brain_mask', nifti_gz_format,
                    't2_2_ref_bet_pipeline'),
        DatasetSpec('t2_2_motion_mats', directory_format,
                    't2_2_motion_mat_pipeline'),
        DatasetSpec('t2_2_dcm_info', text_format,
                    't2_2_dcm_info_pipeline'),
        FieldSpec('t2_2_ped', str, 't2_2_dcm_info_pipeline'),
        FieldSpec('t2_2_pe_angle', str, 't2_2_dcm_info_pipeline'),
        FieldSpec('t2_2_tr', float, 't2_2_dcm_info_pipeline'),
        FieldSpec('t2_2_start_time', str, 't2_2_dcm_info_pipeline'),
        FieldSpec('t2_2_real_duration', str, 't2_2_dcm_info_pipeline'),
        FieldSpec('t2_2_tot_duration', str, 't2_2_dcm_info_pipeline'),
        DatasetSpec('t2_3', dicom_format),
        DatasetSpec('t2_3_reference', nifti_gz_format),
        DatasetSpec('t2_3_nifti', nifti_gz_format, 't2_3_dcm2nii_pipeline'),
        DatasetSpec('t2_3_reg', nifti_gz_format,
                    't2_3_rigid_registration_pipeline'),
        DatasetSpec('t2_3_reg_mat', text_matrix_format,
                    't2_3_rigid_registration_pipeline'),
        DatasetSpec('t2_3_qformed', nifti_gz_format,
                    't2_3_qform_transform_pipeline'),
        DatasetSpec('t2_3_qform_mat', text_matrix_format,
                    't2_3_qform_transform_pipeline'),
        DatasetSpec('t2_3_brain', nifti_gz_format, 't2_3_bet_pipeline'),
        DatasetSpec('t2_3_brain_mask', nifti_gz_format, 't2_3_bet_pipeline'),
        DatasetSpec('t2_3_preproc', nifti_gz_format,
                    't2_3_basic_preproc_pipeline'),
        DatasetSpec('t2_3_ref_preproc', nifti_gz_format,
                    't2_3_ref_basic_preproc_pipeline'),
        DatasetSpec('t2_3_ref_brain', nifti_gz_format, 't2_3_ref_bet_pipeline'),
        DatasetSpec('t2_3_ref_brain_mask', nifti_gz_format,
                    't2_3_ref_bet_pipeline'),
        DatasetSpec('t2_3_motion_mats', directory_format,
                    't2_3_motion_mat_pipeline'),
        DatasetSpec('t2_3_dcm_info', text_format,
                    't2_3_dcm_info_pipeline'),
        FieldSpec('t2_3_ped', str, 't2_3_dcm_info_pipeline'),
        FieldSpec('t2_3_pe_angle', str, 't2_3_dcm_info_pipeline'),
        FieldSpec('t2_3_tr', float, 't2_3_dcm_info_pipeline'),
        FieldSpec('t2_3_start_time', str, 't2_3_dcm_info_pipeline'),
        FieldSpec('t2_3_real_duration', str, 't2_3_dcm_info_pipeline'),
        FieldSpec('t2_3_tot_duration', str, 't2_3_dcm_info_pipeline'),
        DatasetSpec('t2_4', dicom_format),
        DatasetSpec('t2_4_reference', nifti_gz_format),
        DatasetSpec('t2_4_nifti', nifti_gz_format, 't2_4_dcm2nii_pipeline'),
        DatasetSpec('t2_4_reg', nifti_gz_format,
                    't2_4_rigid_registration_pipeline'),
        DatasetSpec('t2_4_reg_mat', text_matrix_format,
                    't2_4_rigid_registration_pipeline'),
        DatasetSpec('t2_4_qformed', nifti_gz_format,
                    't2_4_qform_transform_pipeline'),
        DatasetSpec('t2_4_qform_mat', text_matrix_format,
                    't2_4_qform_transform_pipeline'),
        DatasetSpec('t2_4_brain', nifti_gz_format, 't2_4_bet_pipeline'),
        DatasetSpec('t2_4_brain_mask', nifti_gz_format, 't2_4_bet_pipeline'),
        DatasetSpec('t2_4_preproc', nifti_gz_format,
                    't2_4_basic_preproc_pipeline'),
        DatasetSpec('t2_4_ref_preproc', nifti_gz_format,
                    't2_4_ref_basic_preproc_pipeline'),
        DatasetSpec('t2_4_ref_brain', nifti_gz_format, 't2_4_ref_bet_pipeline'),
        DatasetSpec('t2_4_ref_brain_mask', nifti_gz_format,
                    't2_4_ref_bet_pipeline'),
        DatasetSpec('t2_4_motion_mats', directory_format,
                    't2_4_motion_mat_pipeline'),
        DatasetSpec('t2_4_dcm_info', text_format,
                    't2_4_dcm_info_pipeline'),
        FieldSpec('t2_4_ped', str, 't2_4_dcm_info_pipeline'),
        FieldSpec('t2_4_pe_angle', str, 't2_4_dcm_info_pipeline'),
        FieldSpec('t2_4_tr', float, 't2_4_dcm_info_pipeline'),
        FieldSpec('t2_4_start_time', str, 't2_4_dcm_info_pipeline'),
        FieldSpec('t2_4_real_duration', str, 't2_4_dcm_info_pipeline'),
        FieldSpec('t2_4_tot_duration', str, 't2_4_dcm_info_pipeline'),
        DatasetSpec('t2_5', dicom_format),
        DatasetSpec('t2_5_reference', nifti_gz_format),
        DatasetSpec('t2_5_nifti', nifti_gz_format, 't2_5_dcm2nii_pipeline'),
        DatasetSpec('t2_5_reg', nifti_gz_format,
                    't2_5_rigid_registration_pipeline'),
        DatasetSpec('t2_5_reg_mat', text_matrix_format,
                    't2_5_rigid_registration_pipeline'),
        DatasetSpec('t2_5_qformed', nifti_gz_format,
                    't2_5_qform_transform_pipeline'),
        DatasetSpec('t2_5_qform_mat', text_matrix_format,
                    't2_5_qform_transform_pipeline'),
        DatasetSpec('t2_5_brain', nifti_gz_format, 't2_5_bet_pipeline'),
        DatasetSpec('t2_5_brain_mask', nifti_gz_format, 't2_5_bet_pipeline'),
        DatasetSpec('t2_5_preproc', nifti_gz_format,
                    't2_5_basic_preproc_pipeline'),
        DatasetSpec('t2_5_ref_preproc', nifti_gz_format,
                    't2_5_ref_basic_preproc_pipeline'),
        DatasetSpec('t2_5_ref_brain', nifti_gz_format, 't2_5_ref_bet_pipeline'),
        DatasetSpec('t2_5_ref_brain_mask', nifti_gz_format,
                    't2_5_ref_bet_pipeline'),
        DatasetSpec('t2_5_motion_mats', directory_format,
                    't2_5_motion_mat_pipeline'),
        DatasetSpec('t2_5_dcm_info', text_format,
                    't2_5_dcm_info_pipeline'),
        FieldSpec('t2_5_ped', str, 't2_5_dcm_info_pipeline'),
        FieldSpec('t2_5_pe_angle', str, 't2_5_dcm_info_pipeline'),
        FieldSpec('t2_5_tr', float, 't2_5_dcm_info_pipeline'),
        FieldSpec('t2_5_start_time', str, 't2_5_dcm_info_pipeline'),
        FieldSpec('t2_5_real_duration', str, 't2_5_dcm_info_pipeline'),
        FieldSpec('t2_5_tot_duration', str, 't2_5_dcm_info_pipeline'),
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
        DatasetSpec('mean_displacement', text_format,
                    'mean_displacement_pipeline'),
        DatasetSpec('mean_displacement_rc', text_format,
                    'mean_displacement_pipeline'),
        DatasetSpec('mean_displacement_consecutive', text_format,
                    'mean_displacement_pipeline'),
        DatasetSpec('mats4average', text_format, 'mean_displacement_pipeline'),
        DatasetSpec('start_times', text_format, 'mean_displacement_pipeline'),
        DatasetSpec('motion_par_rc', text_format, 'mean_displacement_pipeline'),
        DatasetSpec('offset_indexes', text_format, 'mean_displacement_pipeline'),
        DatasetSpec('frame_start_times', text_format,
                    'motion_framing_pipeline'),
        DatasetSpec('frame_vol_numbers', text_format,
                    'motion_framing_pipeline'),
        DatasetSpec('mean_displacement_plot', png_format,
                    'plot_mean_displacement_pipeline'),
        DatasetSpec('average_mats', directory_format,
                    'frame_mean_transformation_mats_pipeline'),
        DatasetSpec('correction_factors', text_format,
                    'pet_correction_factors_pipeline'),
        DatasetSpec('umaps_align2ref', directory_format,
                    'frame2ref_alignment_pipeline'),
        DatasetSpec('frame2reference_mats', directory_format,
                    'frame2ref_alignment_pipeline')]
