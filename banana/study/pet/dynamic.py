from .base import PetStudy
from arcana.data import FilesetSpec
from arcana.study.base import StudyMetaClass
from nipype.interfaces.fsl import ExtractROI
from nipype.interfaces.ants.resampling import ApplyTransforms
from arcana.utils.interfaces import Merge
from banana.interfaces.custom.pet import PETdr, GlobalTrendRemoval
from banana.file_format import (nifti_gz_format, png_format)
from arcana.study import ParameterSpec
import os

template_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__).split('arcana')[0],
                 'arcana', 'reference_data'))


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
        ParameterSpec('trans_template',
                      os.path.join(template_path, 'PET_template.nii.gz')),
        ParameterSpec('base_remove_th', 0),
        ParameterSpec('base_remove_binarize', False),
        ParameterSpec('regress_th', 0),
        ParameterSpec('regress_binarize', False)]

    def Extract_vol_pipeline(self, **kwargs):
#             inputs=[FilesetSpec('pet_volumes', nifti_gz_format)],
#             outputs=[FilesetSpec('pet_image', nifti_gz_format)],

        pipeline = self.new_pipeline(
            name='Extract_volume',
            desc=('Extract the last volume of the 4D PET timeseries'),
            references=[],
            **kwargs)

        fslroi = pipeline.add(
            'fslroi',
            ExtractROI(roi_file='vol.nii.gz', t_min=79, t_size=1))
        pipeline.connect_input('pet_volumes', fslroi, 'in_file')
        pipeline.connect_output('pet_image', fslroi, 'roi_file')
        return pipeline

    def ApplyTransform_pipeline(self, **kwargs):
#             inputs=[FilesetSpec('pet_volumes', nifti_gz_format),
#                     FilesetSpec('warp_file', nifti_gz_format),
#                     FilesetSpec('affine_mat', text_matrix_format)],
#             outputs=[FilesetSpec('registered_volumes', nifti_gz_format)],

        pipeline = self.new_pipeline(
            name='applytransform',
            desc=('Apply transformation the the 4D PET timeseries'),
            references=[],
            **kwargs)

        merge_trans = pipeline.add(
            'merge_transforms',
            Merge(2))
        pipeline.connect_input('warp_file', merge_trans, 'in1')
        pipeline.connect_input('affine_mat', merge_trans, 'in2')

        apply_trans = pipeline.add(
            'ApplyTransform',
            ApplyTransforms())
        apply_trans.inputs.reference_image = self.parameter(
            'trans_template')
        apply_trans.inputs.interpolation = 'Linear'
        apply_trans.inputs.input_image_type = 3
        pipeline.connect(merge_trans, 'out', apply_trans, 'transforms')
        pipeline.connect_input('pet_volumes', apply_trans, 'input_image')

        pipeline.connect_output('registered_volumes', apply_trans,
                                'output_image')
        return pipeline

    def Baseline_Removal_pipeline(self, **kwargs):

#             inputs=[FilesetSpec('registered_volumes', nifti_gz_format)],
#             outputs=[FilesetSpec('detrended_volumes', nifti_gz_format)],

        pipeline = self.new_pipeline(
            name='Baseline_removal',
            desc=('PET dual regression'),
            references=[],
            **kwargs)

        br = pipeline.add(
            'Baseline_removal',
            GlobalTrendRemoval())
        pipeline.connect_input('registered_volumes', br, 'volume')
        pipeline.connect_output('detrended_volumes', br, 'detrended_file')
        return pipeline

    def Dual_Regression_pipeline(self, **kwargs):

#             inputs=[FilesetSpec('detrended_volumes', nifti_gz_format),
#                     FilesetSpec('regression_map', nifti_gz_format)],
#             outputs=[FilesetSpec('spatial_map', nifti_gz_format),
#                      FilesetSpec('ts', png_format)],

        pipeline = self.new_pipeline(
            name='Dual_regression',
            desc=('PET dual regression'),
            references=[],
            **kwargs)

        dr = pipeline.add(
            'PET_dr',
            PETdr())
        dr.inputs.threshold = self.parameter('regress_th')
        dr.inputs.binarize = self.parameter('regress_binarize')
        pipeline.connect_input('detrended_volumes', dr, 'volume')
        pipeline.connect_input('regression_map', dr, 'regression_map')

        pipeline.connect_output('spatial_map', dr, 'spatial_map')
        pipeline.connect_output('ts', dr, 'timecourse')
        return pipeline
#     def example_pipeline_switch(self, tool='atool', **kwargs):
#         if tool == 'atool':
#             pipeline = self._atool_pipeline(**kwargs)
#         else:
#             pipeline = self._anothertool_pipeline(**kwargs)
#         return pipeline

    def dynamics_ica_pipeline(self, **kwargs):
        return self._ICA_pipeline_factory(
            input_fileset=FilesetSpec(
                'registered_volumes', nifti_gz_format, **kwargs))
