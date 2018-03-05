from .base import MRIStudy
from nianalysis.dataset import DatasetSpec, FieldSpec
from nianalysis.data_formats import (
    nifti_gz_format, text_matrix_format, directory_format, dicom_format,
    par_format)
from nipype.interfaces.fsl import (ExtractROI, TOPUP, ApplyTOPUP)
from mbianalysis.interfaces.custom.motion_correction import (
    PrepareDWI, CheckDwiNames, GenTopupConfigFiles)
from nianalysis.citations import fsl_cite
from nianalysis.study.base import set_specs
from .coregistered import CoregisteredStudy
from nianalysis.study.multi import MultiStudy
from mbianalysis.interfaces.custom.motion_correction import (
    MotionMatCalculation, AffineMatrixGeneration)
from nianalysis.interfaces.converters import Dcm2niix
from nipype.interfaces.utility import Merge as merge_lists
from mbianalysis.interfaces.mrtrix.preproc import DWIPreproc
from nipype.interfaces.fsl.utils import Merge as fsl_merge
from .epi import CoregisteredEPIStudy


class MotionDetectionStudy(MultiStudy):
    
    sub_study_specs = {
        'epi1': (CoregisteredEPIStudy, {
            'epi1': 'epi',
            'epi1_epireg_mat': 'epi_epireg_mat',
            'epi1_qform_mat': 'epi_qform_mat',
            'epi1_qformed': 'epi_qformed',
            'epi1_moco_mat': 'epi_moco_mat',
            'epi1_motion_mats': 'epi_motion_mats',
            'epi1_preproc': 'epi_preproc'}),
        'epi2': (CoregisteredEPIStudy, {
            'epi2': 'epi',
            'epi2_epireg_mat': 'epi_epireg_mat',
            'epi2_qform_mat': 'epi_qform_mat',
            'epi2_qformed': 'epi_qformed',
            'epi2_moco_mat': 'epi_moco_mat',
            'epi2_motion_mats': 'epi_motion_mats',
            'epi2_preproc': 'epi_preproc'}),
        'reference': (CoregisteredEPIStudy, {
            'reference': 'reference',
            'ref_preproc': 'ref_preproc',
            'ref_brain': 'ref_brain',
            'ref_brain_mask': 'ref_brain_mask',
            'ref_wmseg': 'ref_wmseg'})}
    
    epi1_motion_alignment_pipeline = MultiStudy.translate(
        'epi1', CoregisteredEPIStudy.epi_motion_alignment_pipeline)
    
    epi1_epireg_pipeline = MultiStudy.translate(
        'epi1', CoregisteredEPIStudy.epireg_pipeline)
    
    epi1_motion_mat_pipeline = MultiStudy.translate(
        'epi1', CoregisteredEPIStudy.epi_motion_mat_pipeline)
    
    epi1_basic_preproc_pipeline = MultiStudy.translate(
        'epi1', CoregisteredEPIStudy.epi_basic_preproc_pipeline)
    
    epi1_qform_transform_pipeline = MultiStudy.translate(
        'epi1', CoregisteredEPIStudy.epi_qform_transform_pipeline)
    
    epi2_motion_alignment_pipeline = MultiStudy.translate(
        'epi2', CoregisteredEPIStudy.epi_motion_alignment_pipeline)
    
    epi2_epireg_pipeline = MultiStudy.translate(
        'epi2', CoregisteredEPIStudy.epireg_pipeline)
    
    epi2_motion_mat_pipeline = MultiStudy.translate(
        'epi2', CoregisteredEPIStudy.epi_motion_mat_pipeline)
    
    epi2_basic_preproc_pipeline = MultiStudy.translate(
        'epi2', CoregisteredEPIStudy.epi_basic_preproc_pipeline)
    
    epi2_qform_transform_pipeline = MultiStudy.translate(
        'epi2', CoregisteredEPIStudy.epi_qform_transform_pipeline)
    
    ref_bet_pipeline = MultiStudy.translate(
        'reference', CoregisteredEPIStudy.ref_bet_pipeline)

    ref_segmentation_pipeline = MultiStudy.translate(
        'reference', CoregisteredEPIStudy.ref_segmentation_pipeline)

    ref_basic_preproc_pipeline = MultiStudy.translate(
        'reference', CoregisteredEPIStudy.ref_basic_preproc_pipeline)
    
    _data_specs = set_specs(
        DatasetSpec('epi1', nifti_gz_format),
        DatasetSpec('epi2', nifti_gz_format),
        DatasetSpec('reference', nifti_gz_format),
        DatasetSpec('epi1_preproc', nifti_gz_format,
                    epi1_basic_preproc_pipeline),
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
        DatasetSpec('ref_preproc', nifti_gz_format,
                    ref_basic_preproc_pipeline),
        DatasetSpec('ref_brain', nifti_gz_format, ref_bet_pipeline),
        DatasetSpec('ref_brain_mask', nifti_gz_format,
                    ref_bet_pipeline),
        DatasetSpec('ref_wmseg', nifti_gz_format, ref_segmentation_pipeline))
