#!/usr/bin/env python
from nipype import config
config.enable_debug_mode()
from arcana.data import FilesetSelector  # @IgnorePep8
from banana.study.mri.structural.ute import UTEStudy  # @IgnorePep8
from banana.file_format import (  # @IgnorePep8
    dicom_format, nifti_gz_format, text_matrix_format)
from banana.testing import BaseTestCase as TestCase  # @IgnorePep8


class TestUTE(TestCase):
    '''
    def test_ute(self):
        study = self.create_study(
            UTEStudy, 'registration', {
                FilesetSelector('ute_echo1', 'ute_echo1', dicom_format),
                FilesetSelector('ute_echo2', 'ute_echo2', dicom_format),
                FilesetSelector('umap_ute', 'umap_ute', dicom_format)})
        study.registration_pipeline().run(work_dir=self.work_dir)
        self.assertFilesetCreated('ute1_registered.nii.gz', study.name)
        self.assertFilesetCreated('ute2_registered.nii.gz', study.name)
    
    
    def test_ute(self):
        study = self.create_study(
            UTEStudy, 'segmentation', {
                FilesetSelector('ute1_registered', 'ute1_registered', nifti_gz_format),})
        study.segmentation_pipeline().run(work_dir=self.work_dir)
        self.assertFilesetCreated('air_mask.nii.gz', study.name)
        self.assertFilesetCreated('bones_mask.nii.gz', study.name)
    
    def test_ute(self):
        study = self.create_study(
            UTEStudy, 'umap_creation', {
                FilesetSelector('ute1_registered', 'ute1_registered', nifti_gz_format),
                FilesetSelector('ute2_registered', 'ute2_registered', nifti_gz_format),
                FilesetSelector('air_mask', 'air_mask', nifti_gz_format),
                FilesetSelector('bones_mask', 'bones_mask', nifti_gz_format)})
        study.umaps_calculation_pipeline().run(work_dir=self.work_dir)
        self.assertFilesetCreated('sute_cont_template.nii.gz', study.name)
        self.assertFilesetCreated('sute_fix_template.nii.gz', study.name)
    
    def test_ute(self):
        study = self.create_study(
            UTEStudy, 'backwrap', {
                'ute1_registered':Fileset('ute1_registered', nifti_gz_format),
                'ute_echo1':Fileset('ute_echo1', dicom_format),
                'umap_ute':Fileset('umap_ute', dicom_format),
                'template_to_ute_mat':Fileset('template_to_ute_mat', text_matrix_format),
                'sute_cont_template':Fileset('sute_cont_template', nifti_gz_format),
                'sute_fix_template':Fileset('sute_fix_template', nifti_gz_format)})
        study.backwrap_to_ute_pipeline().run(work_dir=self.work_dir)
        self.assertFilesetCreated('sute_cont_ute.nii.gz', study.name)
        self.assertFilesetCreated('sute_fix_ute.nii.gz', study.name)
    
    
    def test_ute(self):
        study = self.create_study(
            UTEStudy, 'conversion', {
                'sute_cont_ute':Fileset('sute_cont_ute', nifti_gz_format),
                'sute_fix_ute':Fileset('sute_fix_ute', nifti_gz_format),
                FilesetSelector('umap_ute', 'umap_ute', dicom_format)})
        study.conversion_to_dicom_pipeline().run(work_dir=self.work_dir)
        self.assertFilesetCreated('sute_cont_dicoms', study.name)
        self.assertFilesetCreated('sute_fix_dicoms', study.name)
    '''



    def test_ute(self):
        study = self.create_study(
            UTEStudy, 'pipeline', {
                FilesetSelector('ute_echo1', 'ute_echo1', dicom_format),
                FilesetSelector('ute_echo2', 'ute_echo2', dicom_format),
                FilesetSelector('umap_ute', 'umap_ute', dicom_format)})
        study.conversion_to_dicom_pipeline().run(work_dir=self.work_dir)
        self.assertFilesetCreated('sute_cont_dicoms', study.name)
        self.assertFilesetCreated('sute_fix_dicoms', study.name)
    
        
        
        
        
        