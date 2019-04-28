from .base import PetStudy
from arcana.data import FilesetSpec
from arcana.study.base import StudyMetaClass
from nipype.interfaces.fsl import ExtractROI
from nipype.interfaces.ants.resampling import ApplyTransforms
from arcana.utils.interfaces import Merge
from banana.interfaces.custom.pet import PETdr, GlobalTrendRemoval
from banana.file_format import (nifti_gz_format, png_format,
                                text_matrix_format)
from arcana.study import ParamSpec
import os

template_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__).split('arcana')[0],
                 'arcana', 'reference'))


class DynamicPetStudy(PetStudy, metaclass=StudyMetaClass):

    add_data_specs = [
        FilesetSpec('pet_volumes', nifti_gz_format),
        FilesetSpec('regression_map', nifti_gz_format),
        FilesetSpec('pet_image', nifti_gz_format,
                    'Extract_vol_pipeline'),
        FilesetSpec('registered_volumes', nifti_gz_format,
                    'ApplyTransform_pipeline'),
        FilesetSpec('detrended_volumes', nifti_gz_format,
                    'Baseline_Removal_pipeline'),
        FilesetSpec('spatial_map', nifti_gz_format,
                    'Dual_Regression_pipeline'),
        FilesetSpec('ts', png_format, 'Dual_Regression_pipeline')]

    add_param_specs = [
        ParamSpec('trans_template',
                      os.path.join(template_path, 'PET_template.nii.gz')),
        ParamSpec('base_remove_th', 0),
        ParamSpec('base_remove_binarize', False),
        ParamSpec('regress_th', 0),
        ParamSpec('regress_binarize', False)]

    def Extract_vol_pipeline(self, **kwargs):

        pipeline = self.new_pipeline(
            name='Extract_volume',
            desc=('Extract the last volume of the 4D PET timeseries'),
            citations=[],
            **kwargs)

        pipeline.add(
            'fslroi',
            ExtractROI(
                roi_file='vol.nii.gz',
                t_min=79,
                t_size=1),
            inputs={
                'in_file': ('pet_volumes', nifti_gz_format)},
            outputs={
                'pet_image': ('roi_file', nifti_gz_format)})

        return pipeline

    def ApplyTransform_pipeline(self, **kwargs):

        pipeline = self.new_pipeline(
            name='applytransform',
            desc=('Apply transformation the the 4D PET timeseries'),
            citations=[],
            **kwargs)

        merge_trans = pipeline.add(
            'merge_transforms',
            Merge(2),
            inputs={
                'in1': ('warp_file', nifti_gz_format),
                'in2': ('affine_mat', text_matrix_format)})

        pipeline.add(
            'ApplyTransform',
            ApplyTransforms(
                reference_image=self.parameter('trans_template'),
                interpolation='Linear',
                input_image_type=3),
            inputs={
                'input_image': ('pet_volumes', nifti_gz_format),
                'transforms': (merge_trans, 'out')},
            outputs={
                'registered_volumes': ('output_image', nifti_gz_format)})

        return pipeline

    def Baseline_Removal_pipeline(self, **kwargs):

        pipeline = self.new_pipeline(
            name='Baseline_removal',
            desc=('PET dual regression'),
            citations=[],
            **kwargs)

        pipeline.add(
            'Baseline_removal',
            GlobalTrendRemoval(),
            inputs={
                'volume': ('registered_volumes', nifti_gz_format)},
            outputs={
                'detrended_volumes': ('detrended_file', nifti_gz_format)})

        return pipeline

    def Dual_Regression_pipeline(self, **kwargs):

        pipeline = self.new_pipeline(
            name='Dual_regression',
            desc=('PET dual regression'),
            citations=[],
            **kwargs)

        pipeline.add(
            'PET_dr',
            PETdr(
                threshold=self.parameter('regress_th'),
                binarize=self.parameter('regress_binarize')),
            inputs={
                'volume': ('detrended_volumes', nifti_gz_format),
                'regression_map': ('regression_map', nifti_gz_format)},
            outputs={
                'spatial_map': ('spatial_map', nifti_gz_format),
                'ts': ('timecourse', png_format)})

        return pipeline

    def dynamics_ica_pipeline(self, **kwargs):
        return self._ICA_pipeline_factory(
            input_fileset=FilesetSpec(
                'registered_volumes', nifti_gz_format, **kwargs))
