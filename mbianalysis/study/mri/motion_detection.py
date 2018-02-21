from .base import MRIStudy
from nianalysis.dataset import DatasetSpec, FieldSpec
from nianalysis.data_formats import (
    nifti_gz_format, text_matrix_format, directory_format, dicom_format,
    par_format, text_format)
from nipype.interfaces.fsl import (ExtractROI, TOPUP, ApplyTOPUP)
from mbianalysis.interfaces.custom.motion_correction import (
    MeanDisplacementCalculation)
from nianalysis.citations import fsl_cite
from nianalysis.study.base import set_data_specs
from .coregistered import CoregisteredStudy
from nianalysis.study.combined import CombinedStudy
from mbianalysis.interfaces.custom.dicom import ScanTimesInfo
from .epi import CoregisteredEPIStudy
from nipype.interfaces.utility import Merge as merge_lists
from nianalysis.interfaces.converters import Dcm2niix


class MotionDetectionStudy(CombinedStudy):

    sub_study_specs = {
        'epi1': (CoregisteredEPIStudy, {
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
        'epi2': (CoregisteredEPIStudy, {
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
            'epi2_dcm_info': 'epi_dcm_info'})}

    epi1_motion_alignment_pipeline = CombinedStudy.translate(
        'epi1', CoregisteredEPIStudy.epi_motion_alignment_pipeline)

    epi1_dcm2nii_pipeline = CombinedStudy.translate(
        'epi1', CoregisteredEPIStudy.epi_dcm2nii_pipeline)

    epi1_epireg_pipeline = CombinedStudy.translate(
        'epi1', CoregisteredEPIStudy.epireg_pipeline)

    epi1_dcm_info_pipeline = CombinedStudy.translate(
        'epi1', CoregisteredEPIStudy.epi_dcm_info_pipeline)

    epi1_motion_mat_pipeline = CombinedStudy.translate(
        'epi1', CoregisteredEPIStudy.epi_motion_mat_pipeline)

    epi1_basic_preproc_pipeline = CombinedStudy.translate(
        'epi1', CoregisteredEPIStudy.epi_basic_preproc_pipeline)

    epi1_qform_transform_pipeline = CombinedStudy.translate(
        'epi1', CoregisteredEPIStudy.epi_qform_transform_pipeline)

    epi1_bet_pipeline = CombinedStudy.translate(
        'epi1', CoregisteredEPIStudy.epi_bet_pipeline)

    epi2_motion_alignment_pipeline = CombinedStudy.translate(
        'epi2', CoregisteredEPIStudy.epi_motion_alignment_pipeline)

    epi2_dcm_info_pipeline = CombinedStudy.translate(
        'epi2', CoregisteredEPIStudy.epi_dcm_info_pipeline)

    epi2_epireg_pipeline = CombinedStudy.translate(
        'epi2', CoregisteredEPIStudy.epireg_pipeline)

    epi2_motion_mat_pipeline = CombinedStudy.translate(
        'epi2', CoregisteredEPIStudy.epi_motion_mat_pipeline)

    epi2_basic_preproc_pipeline = CombinedStudy.translate(
        'epi2', CoregisteredEPIStudy.epi_basic_preproc_pipeline)

    epi2_qform_transform_pipeline = CombinedStudy.translate(
        'epi2', CoregisteredEPIStudy.epi_qform_transform_pipeline)

    epi1_ref_bet_pipeline = CombinedStudy.translate(
        'epi1', CoregisteredEPIStudy.ref_bet_pipeline)

    epi1_ref_segmentation_pipeline = CombinedStudy.translate(
        'epi1', CoregisteredEPIStudy.ref_segmentation_pipeline)

    epi1_ref_basic_preproc_pipeline = CombinedStudy.translate(
        'epi1', CoregisteredEPIStudy.ref_basic_preproc_pipeline)

    epi1_ref_nifti_pipeline = CombinedStudy.translate(
        'epi1', CoregisteredEPIStudy.ref_dcm2nii_pipeline)

    epi2_ref_bet_pipeline = CombinedStudy.translate(
        'epi2', CoregisteredEPIStudy.ref_bet_pipeline)

    epi2_ref_nifti_pipeline = CombinedStudy.translate(
        'epi2', CoregisteredEPIStudy.ref_dcm2nii_pipeline)

    epi2_ref_segmentation_pipeline = CombinedStudy.translate(
        'epi2', CoregisteredEPIStudy.ref_segmentation_pipeline)

    epi2_ref_basic_preproc_pipeline = CombinedStudy.translate(
        'epi2', CoregisteredEPIStudy.ref_basic_preproc_pipeline)

    epi2_bet_pipeline = CombinedStudy.translate(
        'epi2', CoregisteredEPIStudy.epi_bet_pipeline)

    epi2_dcm2nii_pipeline = CombinedStudy.translate(
        'epi2', CoregisteredEPIStudy.epi_dcm2nii_pipeline)

    def scans_time_info_pipeline(self, **options):

        pipeline = self.create_pipeline(
            name='scan_time_information',
            inputs=[DatasetSpec('epi1_dcm_info', text_format),
                    DatasetSpec('epi2_dcm_info', text_format)],
            outputs=[DatasetSpec('time_infos', text_format)],
            description=("Extract time information from all the scans."),
            default_options={},
            version=1,
            citations=[fsl_cite],
            options=options)

        merge = pipeline.create_node(merge_lists(2), name='merge_inputs')
        pipeline.connect_input('epi1_dcm_info', merge, 'in1')
        pipeline.connect_input('epi2_dcm_info', merge, 'in2')

        time_info = pipeline.create_node(ScanTimesInfo(),
                                         name='scan_time_info')
        pipeline.connect(merge, 'out', time_info, 'dicom_infos')
        pipeline.connect_output('time_infos', time_info, 'scan_time_infos')
        pipeline.assert_connected()
        return pipeline

    def mean_displacement_pipeline(self, **options):

        pipeline = self.create_pipeline(
            name='mean_displacement_calculation',
            inputs=[DatasetSpec('epi1_motion_mats', directory_format),
                    DatasetSpec('epi2_motion_mats', directory_format),
                    DatasetSpec('epi1_reference', nifti_gz_format),
                    FieldSpec('epi1_tr', float),
                    FieldSpec('epi1_start_time', str),
                    FieldSpec('epi1_real_duration', str),
                    FieldSpec('epi2_tr', float),
                    FieldSpec('epi2_start_time', str),
                    FieldSpec('epi2_real_duration', str)],
            outputs=[DatasetSpec('mean_displacement', text_format),
                     DatasetSpec('mean_displacement_rc', text_format),
                     DatasetSpec('mean_displacement_consecutive', text_format),
                     DatasetSpec('start_times', text_format),
                     DatasetSpec('motion_par_rc', text_format)],
            description=("Calculate the mean displacement between each motion"
                         " matrix and a reference."),
            default_options={},
            version=1,
            citations=[fsl_cite],
            options=options)

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

        merge_scans = pipeline.create_node(merge_lists(2), name='merge_scans')
        merge_scans.inputs.no_flatten = True
        pipeline.connect(merge_epi1, 'out', merge_scans, 'in1')
        pipeline.connect(merge_epi2, 'out', merge_scans, 'in2')

        md = pipeline.create_node(MeanDisplacementCalculation(),
                                  name='scan_time_info')
        pipeline.connect(merge_scans, 'out', md, 'list_inputs')
        pipeline.connect_input('epi1_reference', md, 'reference')
        pipeline.connect_output('mean_displacement', md, 'mean_displacement')
        pipeline.connect_output(
            'mean_displacement_rc', md, 'mean_displacement_rc')
        pipeline.connect_output(
            'mean_displacement_consecutive', md,
            'mean_displacement_consecutive')
        pipeline.connect_output('start_times', md, 'start_times')
        pipeline.connect_output('motion_par_rc', md, 'motion_parameters')
        pipeline.assert_connected()
        return pipeline

    _data_specs = set_data_specs(
        DatasetSpec('epi1', dicom_format),
        DatasetSpec('epi2', dicom_format),
        DatasetSpec('epi1_reference', nifti_gz_format),
        DatasetSpec('epi1_nifti', nifti_gz_format, epi1_dcm2nii_pipeline),
        DatasetSpec('epi2_nifti', nifti_gz_format, epi2_dcm2nii_pipeline),
        DatasetSpec('epi2_reference', nifti_gz_format),
        DatasetSpec('epi1_preproc', nifti_gz_format,
                    epi1_basic_preproc_pipeline),
        DatasetSpec('epi1_brain', nifti_gz_format,
                    epi1_bet_pipeline),
        DatasetSpec('epi1_brain_mask', nifti_gz_format,
                    epi1_bet_pipeline),
        DatasetSpec('epi1_qformed', nifti_gz_format,
                    epi1_qform_transform_pipeline),
        DatasetSpec('epi1_qform_mat', text_matrix_format,
                    epi1_qform_transform_pipeline),
        DatasetSpec('epi1_epireg', nifti_gz_format, epi1_epireg_pipeline),
        DatasetSpec('epi1_epireg_mat', text_matrix_format,
                    epi1_epireg_pipeline),
        DatasetSpec('epi1_motion_mats', directory_format,
                    epi1_motion_mat_pipeline),
        DatasetSpec('epi1_moco', nifti_gz_format,
                    epi1_motion_alignment_pipeline),
        DatasetSpec('epi1_moco_mat', directory_format,
                    epi1_motion_alignment_pipeline),
        DatasetSpec('epi1_moco_par', par_format,
                    epi1_motion_alignment_pipeline),
        DatasetSpec('epi2_preproc', nifti_gz_format,
                    epi2_basic_preproc_pipeline),
        DatasetSpec('epi2_qformed', nifti_gz_format,
                    epi2_qform_transform_pipeline),
        DatasetSpec('epi2_qform_mat', text_matrix_format,
                    epi2_qform_transform_pipeline),
        DatasetSpec('epi2_epireg', nifti_gz_format, epi2_epireg_pipeline),
        DatasetSpec('epi2_epireg_mat', text_matrix_format,
                    epi2_epireg_pipeline),
        DatasetSpec('epi2_motion_mats', directory_format,
                    epi2_motion_mat_pipeline),
        DatasetSpec('epi2_moco', nifti_gz_format,
                    epi2_motion_alignment_pipeline),
        DatasetSpec('epi2_moco_mat', directory_format,
                    epi2_motion_alignment_pipeline),
        DatasetSpec('epi2_moco_par', par_format,
                    epi2_motion_alignment_pipeline),
        DatasetSpec('epi2_brain', nifti_gz_format,
                    epi2_bet_pipeline),
        DatasetSpec('epi2_brain_mask', nifti_gz_format,
                    epi2_bet_pipeline),
        DatasetSpec('epi1_ref_preproc', nifti_gz_format,
                    epi1_ref_basic_preproc_pipeline),
        DatasetSpec('epi1_ref_brain', nifti_gz_format, epi1_ref_bet_pipeline),
        DatasetSpec('epi1_ref_brain_mask', nifti_gz_format,
                    epi1_ref_bet_pipeline),
        DatasetSpec(
            'epi1_ref_wmseg',
            nifti_gz_format,
            epi1_ref_segmentation_pipeline),
        DatasetSpec('epi2_ref_preproc', nifti_gz_format,
                    epi2_ref_basic_preproc_pipeline),
        DatasetSpec('epi2_ref_brain', nifti_gz_format, epi2_ref_bet_pipeline),
        DatasetSpec('epi2_ref_brain_mask', nifti_gz_format,
                    epi2_ref_bet_pipeline),
        DatasetSpec('epi2_ref_wmseg', nifti_gz_format,
                    epi2_ref_segmentation_pipeline),
        DatasetSpec('epi1_dcm_info', text_format,
                    epi1_dcm_info_pipeline),
        DatasetSpec('epi2_dcm_info', text_format,
                    epi2_dcm_info_pipeline),
        FieldSpec('epi1_ped', str, epi1_dcm_info_pipeline),
        FieldSpec('epi1_pe_angle', str, epi1_dcm_info_pipeline),
        FieldSpec('epi1_tr', float, epi1_dcm_info_pipeline),
        FieldSpec('epi1_start_time', str, epi1_dcm_info_pipeline),
        FieldSpec('epi1_real_duration', str, epi1_dcm_info_pipeline),
        FieldSpec('epi1_tot_duration', str, epi1_dcm_info_pipeline),
        FieldSpec('epi2_ped', str, epi2_dcm_info_pipeline),
        FieldSpec('epi2_pe_angle', str, epi2_dcm_info_pipeline),
        FieldSpec('epi2_tr', float, epi2_dcm_info_pipeline),
        FieldSpec('epi2_start_time', str, epi2_dcm_info_pipeline),
        FieldSpec('epi2_real_duration', str, epi2_dcm_info_pipeline),
        FieldSpec('epi2_tot_duration', str, epi2_dcm_info_pipeline),
        DatasetSpec('time_infos', text_format, scans_time_info_pipeline),
        DatasetSpec('mean_displacement', text_format,
                    mean_displacement_pipeline),
        DatasetSpec('mean_displacement_rc', text_format,
                    mean_displacement_pipeline),
        DatasetSpec('mean_displacement_consecutive', text_format,
                    mean_displacement_pipeline),
        DatasetSpec('start_times', text_format, mean_displacement_pipeline),
        DatasetSpec('motion_par_rc', text_format, mean_displacement_pipeline))
