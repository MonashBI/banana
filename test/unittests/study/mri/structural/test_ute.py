#!/usr/bin/env python
from nipype import config
config.enable_debug_mode()
from nianalysis.dataset import Dataset  # @IgnorePep8
from nianalysis.study.mri.structural.ute import UTEStudy
from nianalysis.data_formats import (  # @IgnorePep8
    dicom_format, nifti_gz_format, text_matrix_format)
from nianalysis.testing import BaseTestCase as TestCase  # @IgnorePep8 @Reimport


class TestUTE(TestCase):
    '''
    def test_ute(self):
        study = self.create_study(
            UTEStudy, 'registration', {
                'ute_echo1': Dataset('ute_echo1', dicom_format),
                'ute_echo2': Dataset('ute_echo2', dicom_format),
                'umap_ute': Dataset('umap_ute', dicom_format)})
        study.registration_pipeline().run(work_dir=self.work_dir)
        self.assertDatasetCreated('ute1_registered.nii.gz', study.name)
        self.assertDatasetCreated('ute2_registered.nii.gz', study.name)
     
       
    def test_ute(self):
        study = self.create_study(
            UTEStudy, 'segmentation', {
                'ute_echo1': Dataset('ute_echo1', dicom_format),
                'ute_echo2': Dataset('ute_echo2', dicom_format),
                'umap_ute': Dataset('umap_ute', dicom_format)})
        study.umaps_calculation_pipeline().run(work_dir=self.work_dir)
        self.assertDatasetCreated('sute_cont.nii.gz', study.name)
        self.assertDatasetCreated('sute_fix.nii.gz', study.name)
    
    
    def test_ute(self):
        study = self.create_study(
            UTEStudy, 'umap_creation', {
                'ute1_registered': Dataset('ute1_registered', nifti_gz_format),
                'ute2_registered': Dataset('ute2_registered', nifti_gz_format),
                'air_mask': Dataset('air_mask', nifti_gz_format),
                'bones_mask': Dataset('bones_mask', nifti_gz_format)})
        study.umaps_calculation_pipeline().run(work_dir=self.work_dir)
        self.assertDatasetCreated('sute_cont_template.nii.gz', study.name)
        self.assertDatasetCreated('sute_fix_template.nii.gz', study.name)  
        
    def test_ute(self):
        study = self.create_study(
            UTEStudy, 'umap_creation_ute', {
                'ute1_registered':Dataset('ute1_registered', nifti_gz_format),
                'ute_echo1':Dataset('ute_echo1', dicom_format),
                'umap_ute':Dataset('umap_ute', dicom_format),
                'template_to_ute_mat':Dataset('template_to_ute_mat', text_matrix_format),
                'sute_cont_template':Dataset('sute_cont_template', nifti_gz_format),
                'sute_fix_template':Dataset('sute_fix_template', nifti_gz_format)})
        study.backwrap_to_ute_pipeline().run(work_dir=self.work_dir)
        self.assertDatasetCreated('sute_cont_ute.nii.gz', study.name)
        self.assertDatasetCreated('sute_fix_ute.nii.gz', study.name)
        self.assertDatasetCreated('sute_cont_ute_background.nii.gz', study.name)
        self.assertDatasetCreated('sute_fix_ute_background.nii.gz', study.name)
    '''    
        
        
    def test_ute(self):
        study = self.create_study(
            UTEStudy, 'conversion', {
                'sute_cont_ute':Dataset('sute_cont_ute', nifti_gz_format),
                'sute_fix_ute':Dataset('sute_fix_ute', nifti_gz_format),
                'umap_ute':Dataset('umap_ute', dicom_format)})
        study.conversion_to_dicom_pipeline().run(work_dir=self.work_dir)
        self.assertDatasetCreated('sute_cont_dicoms', study.name)
        self.assertDatasetCreated('sute_fix_dicoms', study.name)
        
        