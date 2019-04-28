from arcana.study.base import Study, StudyMetaClass
from arcana.data import FilesetSpec, FieldSpec, InputFilesetSpec
from banana.file_format import (
    nifti_gz_format, text_format, text_matrix_format, directory_format,
    list_mode_format)
from banana.interfaces.sklearn import FastICA
from banana.interfaces.ants import AntsRegSyn
import os
from banana.requirement import fsl_req, mrtrix_req
from banana.interfaces.custom.pet import PreparePetDir
from banana.interfaces.custom.dicom import PetTimeInfo
from arcana.study import ParamSpec
from banana.interfaces.custom.pet import (
    PrepareUnlistingInputs, PETListModeUnlisting, SSRB, MergeUnlistingOutputs)
from banana.requirement import stir_req


template_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__).split('arcana')[0],
                 'arcana', 'reference'))


class PetStudy(Study, metaclass=StudyMetaClass):

    add_param_specs = [ParamSpec('ica_n_components', 2),
                        ParamSpec('ica_type', 'spatial'),
                        ParamSpec('norm_transformation', 's'),
                        ParamSpec('norm_dim', 3),
                        ParamSpec('norm_template',
                                      os.path.join(template_path,
                                                   'PET_template.nii.gz')),
                        ParamSpec('crop_xmin', 100),
                        ParamSpec('crop_xsize', 130),
                        ParamSpec('crop_ymin', 100),
                        ParamSpec('crop_ysize', 130),
                        ParamSpec('crop_zmin', 20),
                        ParamSpec('crop_zsize', 100),
                        ParamSpec('image_orientation_check', False)]

    add_data_specs = [
        InputFilesetSpec('list_mode', list_mode_format),
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
                  'pet_time_info_extraction_pipeline'),
        FieldSpec('pet_end_time', dtype=str,
                  'pet_time_info_extraction_pipeline'),
        FieldSpec('pet_start_time', dtype=str,
                  'pet_time_info_extraction_pipeline'),
        FieldSpec('time_offset', int),
        FieldSpec('temporal_length', float),
        FieldSpec('num_frames', int),
        FilesetSpec('ssrb_sinograms', directory_format,
                    'sinogram_unlisting_pipeline')]

    def ICA_pipeline(self, **kwargs):

        pipeline = self.new_pipeline(
            name='ICA',
            desc=('Decompose a 4D fileset into a set of independent '
                  'components using FastICA'),
            citations=[],
            **kwargs)

        pipeline.add(
            'ICA',
            FastICA(
                n_components=self.parameter('ica_n_components'),
                ica_type=self.parameter('ica_type')),
            inputs={
                'volume': ('registered_volumes', nifti_gz_format)},
            ouputs={
                'decomposed_file': ('ica_decomposition', nifti_gz_format),
                'timeseries': ('ica_timeseries', nifti_gz_format),
                'mixing_mat': ('mixing_mat', text_format)})

        return pipeline

    def Image_normalization_pipeline(self, **kwargs):

        pipeline = self.new_pipeline(
            name='Image_registration',
            desc=('Image registration to a template using ANTs'),
            citations=[],
            **kwargs)

        pipeline.add(
            'ANTs',
            AntsRegSyn(
                out_prefix='vol2template',
                num_dimensions=self.parameter('norm_dim'),
                num_threads=self.processor.num_processes,
                transformation=self.parameter('norm_transformation'),
                ref_file=self.parameter('norm_template')),
            inputs={
                'input_file': ('pet_image', nifti_gz_format)},
            ouputs={
                'registered_volume': ('reg_file', nifti_gz_format),
                'warp_file': ('warp_file', nifti_gz_format),
                'invwarp_file': ('inv_warp', nifti_gz_format),
                'affine_mat': ('regmat', text_matrix_format)})

        return pipeline

    def pet_data_preparation_pipeline(self, **kwargs):

        pipeline = self.new_pipeline(
            name='pet_data_preparation',
            desc=("Given a folder with reconstructed PET data, this "
                  "pipeline will prepare the data for the motion "
                  "correction"),
            citations=[],
            **kwargs)

        pipeline.add(
            'prepare_pet',
            PreparePetDir(
                image_orientation_check=self.parameter(
                    'image_orientation_check')),
            inputs={
                'pet_dir': ('pet_recon_dir', directory_format)},
            ouputs={
                'pet_recon_dir_prepared': ('pet_dir_prepared',
                                           directory_format)},
            requirements=[mrtrix_req.v('3.0rc3'), fsl_req.v('5.0.9')])

        return pipeline

    def pet_time_info_extraction_pipeline(self, **kwargs):

        pipeline = self.new_pipeline(
            name='pet_info_extraction',
            desc=("Extract PET time info from list-mode header."),
            citations=[],
            **kwargs)

        pipeline.add(
            'PET_time_info',
            PetTimeInfo(),
            inputs={
                'pet_data_dir': ('pet_data_dir', directory_format)},
            ouputs={
                'pet_end_time': ('pet_end_time', float),
                'pet_start_time': ('pet_start_time', str),
                'pet_duration': ('pet_duration', int)})
        return pipeline

    def sinogram_unlisting_pipeline(self, **kwargs):

        pipeline = self.new_pipeline(
            name='prepare_sinogram',
            desc=('Unlist pet listmode data into several sinograms and '
                         'perform ssrb compression to prepare data for motion '
                         'detection using PCA pipeline.'),
            citations=[],
            **kwargs)

        prepare_inputs = pipeline.add(
            'prepare_inputs',
            PrepareUnlistingInputs(),
            inputs={
                'list_mode': ('list_mode', list_mode_format),
                'time_offset': ('time_offset', int),
                'num_frames': ('num_frames', int),
                'temporal_len': ('temporal_length', float)})

        unlisting = pipeline.add(
            'unlisting',
            PETListModeUnlisting(),
            inputs={
                'list_inputs': (prepare_inputs, 'out')},
            iterfield=['list_inputs'])

        ssrb = pipeline.add(
            'ssrb',
            SSRB(),
            inputs={
                'unlisted_sinogram': (unlisting, 'pet_sinogram')},
            requirements=[stir_req.v('3.0')])

        pipeline.add(
            'merge_sinograms',
            MergeUnlistingOutputs(),
            inputs={
                'sinograms': (ssrb, 'ssrb_sinograms')},
            ouputs={
                'ssrb_sinograms': ('sinogram_folder', directory_format)},
            joinsource='unlisting',
            joinfield=['sinograms'])

        return pipeline
