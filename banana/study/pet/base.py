from arcana.study.base import Study, StudyMetaClass
from arcana.data import FilesetSpec, FieldSpec
from banana.file_format import (nifti_gz_format, text_format,
                                    text_matrix_format, directory_format)
from banana.interfaces.sklearn import FastICA
from banana.interfaces.ants import AntsRegSyn
import os
from banana.requirement import fsl_req, mrtrix_req
from banana.interfaces.custom.pet import PreparePetDir
from banana.interfaces.custom.dicom import PetTimeInfo
from arcana.study import ParameterSpec


template_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__).split('arcana')[0],
                 'arcana', 'reference_data'))


class PetStudy(Study, metaclass=StudyMetaClass):

    add_param_specs = [ParameterSpec('ica_n_components', 2),
                        ParameterSpec('ica_type', 'spatial'),
                        ParameterSpec('norm_transformation', 's'),
                        ParameterSpec('norm_dim', 3),
                        ParameterSpec('norm_template',
                                      os.path.join(template_path,
                                                   'PET_template.nii.gz')),
                        ParameterSpec('crop_xmin', 100),
                        ParameterSpec('crop_xsize', 130),
                        ParameterSpec('crop_ymin', 100),
                        ParameterSpec('crop_ysize', 130),
                        ParameterSpec('crop_zmin', 20),
                        ParameterSpec('crop_zsize', 100),
                        ParameterSpec('image_orientation_check', False)]

    add_data_specs = [
        FilesetSpec('registered_volumes', nifti_gz_format, optional=True),
        FilesetSpec('pet_image', nifti_gz_format, optional=True),
        FilesetSpec('pet_data_dir', directory_format),
        FilesetSpec('pet_recon_dir', directory_format),
        FilesetSpec('pet_recon_dir_prepared', directory_format,
                    'pet_data_preparation_pipeline'),
        FilesetSpec('decomposed_file', nifti_gz_format, 'ICA_pipeline'),
        FilesetSpec('timeseries', nifti_gz_format, 'ICA_pipeline'),
        FilesetSpec('mixing_mat', text_format, 'ICA_pipeline'),
        FilesetSpec('registered_volume', nifti_gz_format,
                    'Image_normalization_pipeline'),
        FilesetSpec('warp_file', nifti_gz_format,
                    'Image_normalization_pipeline'),
        FilesetSpec('invwarp_file', nifti_gz_format,
                    'Image_normalization_pipeline'),
        FilesetSpec('affine_mat', text_matrix_format,
                    'Image_normalization_pipeline'),
        FieldSpec('pet_duration', dtype=int,
                  pipeline_name='pet_time_info_extraction_pipeline'),
        FieldSpec('pet_end_time', dtype=str,
                  pipeline_name='pet_time_info_extraction_pipeline'),
        FieldSpec('pet_start_time', dtype=str,
                  pipeline_name='pet_time_info_extraction_pipeline')]

    def ICA_pipeline(self, **kwargs):
        return self._ICA_pipeline_factory(
            input_fileset=FilesetSpec('registered_volumes', nifti_gz_format),
            **kwargs)

    def _ICA_pipeline_factory(self, input_fileset, **kwargs):

#             inputs=[input_fileset],
#             outputs=[FilesetSpec('decomposed_file', nifti_gz_format),
#                      FilesetSpec('timeseries', nifti_gz_format),
#                      FilesetSpec('mixing_mat', text_format)],


        pipeline = self.new_pipeline(
            name='ICA',
            desc=('Decompose a 4D fileset into a set of independent '
                  'components using FastICA'),
            references=[],
            **kwargs)

        ica = pipeline.add(
            'ICA',
            FastICA())
        ica.inputs.n_components = self.parameter('ica_n_components')
        ica.inputs.ica_type = self.parameter('ica_type')
        pipeline.connect_input('registered_volumes', ica, 'volume')

        pipeline.connect_output('decomposed_file', ica, 'ica_decomposition')
        pipeline.connect_output('timeseries', ica, 'ica_timeseries')
        pipeline.connect_output('mixing_mat', ica, 'mixing_mat')

        return pipeline

    def Image_normalization_pipeline(self, **kwargs):


#             inputs=[FilesetSpec('pet_image', nifti_gz_format)],
#             outputs=[FilesetSpec('registered_volume', nifti_gz_format),
#                      FilesetSpec('warp_file', nifti_gz_format),
#                      FilesetSpec('invwarp_file', nifti_gz_format),
#                      FilesetSpec('affine_mat', text_matrix_format)],

        pipeline = self.new_pipeline(
            name='Image_registration',
            desc=('Image registration to a template using ANTs'),
            references=[],
            **kwargs)

        reg = pipeline.add(
            'ANTs',
            AntsRegSyn(out_prefix='vol2template'))
        reg.inputs.num_dimensions = self.parameter('norm_dim')
        reg.inputs.num_threads = self.processor.num_processes
        reg.inputs.transformation = self.parameter('norm_transformation')
        reg.inputs.ref_file = self.parameter('norm_template')
        pipeline.connect_input('pet_image', reg, 'input_file')

        pipeline.connect_output('registered_volume', reg, 'reg_file')
        pipeline.connect_output('warp_file', reg, 'warp_file')
        pipeline.connect_output('invwarp_file', reg, 'inv_warp')
        pipeline.connect_output('affine_mat', reg, 'regmat')
        return pipeline

    def pet_data_preparation_pipeline(self, **kwargs):

#             inputs=[FilesetSpec('pet_recon_dir', directory_format)],
#             outputs=[FilesetSpec('pet_recon_dir_prepared', directory_format)],


        pipeline = self.new_pipeline(
            name='pet_data_preparation',
            desc=("Given a folder with reconstructed PET data, this "
                  "pipeline will prepare the data for the motion "
                  "correction"),
            references=[],
            **kwargs)

        prep_dir = pipeline.add(
            'prepare_pet',
            PreparePetDir(),
            requirements=[mrtrix_req.v('3.0rc3'), fsl_req.v('5.0.9')])
        prep_dir.inputs.image_orientation_check = self.parameter(
            'image_orientation_check')
        pipeline.connect_input('pet_recon_dir', prep_dir, 'pet_dir')

        pipeline.connect_output('pet_recon_dir_prepared', prep_dir,
                                'pet_dir_prepared')
        return pipeline

    def pet_time_info_extraction_pipeline(self, **kwargs):
#             inputs=[FilesetSpec('pet_data_dir', directory_format)],
#             outputs=[FieldSpec('pet_end_time', dtype=float),
#                      FieldSpec('pet_start_time', dtype=str),
#                      FieldSpec('pet_duration', dtype=int)],

        
        pipeline = self.new_pipeline(
            name='pet_info_extraction',
            desc=("Extract PET time info from list-mode header."),
            references=[],
            **kwargs)
        time_info = pipeline.add(
            'PET_time_info',
            PetTimeInfo())
        pipeline.connect_input('pet_data_dir', time_info, 'pet_data_dir')
        pipeline.connect_output('pet_end_time', time_info, 'pet_end_time')
        pipeline.connect_output('pet_start_time', time_info, 'pet_start_time')
        pipeline.connect_output('pet_duration', time_info, 'pet_duration')
        return pipeline
