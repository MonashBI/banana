#!/usr/bin/env python
from nipype import config
config.enable_debug_mode()
from arcana.dataset import DatasetMatch  # @IgnorePep8
from mbianalysis.study.mri.structural.ute import UTEStudy  # @IgnorePep8
from mbianalysis.data_format import (  # @IgnorePep8
    dicom_format, nifti_gz_format, text_matrix_format)
from mbianalysis.testing import BaseTestCase as TestCase  # @IgnorePep8


class TestUTE(TestCase):
    '''
    def test_ute(self):
        study = self.create_study(
            UTEStudy, 'registration', {
                DatasetMatch('ute_echo1', dicom_format, 'ute_echo1'),
                DatasetMatch('ute_echo2', dicom_format, 'ute_echo2'),
                DatasetMatch('umap_ute', dicom_format, 'umap_ute')})
        study.registration_pipeline().run(work_dir=self.work_dir)
        self.assertDatasetCreated('ute1_registered.nii.gz', study.name)
        self.assertDatasetCreated('ute2_registered.nii.gz', study.name)
    
    
    def test_ute(self):
        study = self.create_study(
            UTEStudy, 'segmentation', {
                DatasetMatch('ute1_registered', nifti_gz_format, 'ute1_registered'),})
        study.segmentation_pipeline().run(work_dir=self.work_dir)
        self.assertDatasetCreated('air_mask.nii.gz', study.name)
        self.assertDatasetCreated('bones_mask.nii.gz', study.name)
    
    def test_ute(self):
        study = self.create_study(
            UTEStudy, 'umap_creation', {
                DatasetMatch('ute1_registered', nifti_gz_format, 'ute1_registered'),
                DatasetMatch('ute2_registered', nifti_gz_format, 'ute2_registered'),
                DatasetMatch('air_mask', nifti_gz_format, 'air_mask'),
                DatasetMatch('bones_mask', nifti_gz_format, 'bones_mask')})
        study.umaps_calculation_pipeline().run(work_dir=self.work_dir)
        self.assertDatasetCreated('sute_cont_template.nii.gz', study.name)
        self.assertDatasetCreated('sute_fix_template.nii.gz', study.name)
    
    def test_ute(self):
        study = self.create_study(
            UTEStudy, 'backwrap', {
                'ute1_registered':Dataset('ute1_registered', nifti_gz_format),
                'ute_echo1':Dataset('ute_echo1', dicom_format),
                'umap_ute':Dataset('umap_ute', dicom_format),
                'template_to_ute_mat':Dataset('template_to_ute_mat', text_matrix_format),
                'sute_cont_template':Dataset('sute_cont_template', nifti_gz_format),
                'sute_fix_template':Dataset('sute_fix_template', nifti_gz_format)})
        study.backwrap_to_ute_pipeline().run(work_dir=self.work_dir)
        self.assertDatasetCreated('sute_cont_ute.nii.gz', study.name)
        self.assertDatasetCreated('sute_fix_ute.nii.gz', study.name)
    
    
    def test_ute(self):
        study = self.create_study(
            UTEStudy, 'conversion', {
                'sute_cont_ute':Dataset('sute_cont_ute', nifti_gz_format),
                'sute_fix_ute':Dataset('sute_fix_ute', nifti_gz_format),
                DatasetMatch('umap_ute', dicom_format, 'umap_ute')})
        study.conversion_to_dicom_pipeline().run(work_dir=self.work_dir)
        self.assertDatasetCreated('sute_cont_dicoms', study.name)
        self.assertDatasetCreated('sute_fix_dicoms', study.name)
    '''



    def test_ute(self):
        study = self.create_study(
            UTEStudy, 'pipeline', {
                DatasetMatch('ute_echo1', dicom_format, 'ute_echo1'),
                DatasetMatch('ute_echo2', dicom_format, 'ute_echo2'),
                DatasetMatch('umap_ute', dicom_format, 'umap_ute')})
        study.conversion_to_dicom_pipeline().run(work_dir=self.work_dir)
        self.assertDatasetCreated('sute_cont_dicoms', study.name)
        self.assertDatasetCreated('sute_fix_dicoms', study.name)
    
        
        
        
        
        