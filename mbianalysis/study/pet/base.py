from nianalysis.study.base import Study, set_specs
from nianalysis.dataset import DatasetSpec, FieldSpec
from nianalysis.data_formats import (nifti_gz_format, text_format,
                                     text_matrix_format, directory_format)
from mbianalysis.interfaces.sklearn import FastICA
from mbianalysis.interfaces.ants import AntsRegSyn
import os
from abc import abstractmethod
from nianalysis.requirements import (fsl5_req, ants2_req, afni_req, fix_req,
                                     fsl509_req, fsl510_req, mrtrix3_req)
from nianalysis.citations import fsl_cite
from mbianalysis.interfaces.custom.pet import PETFovCropping, PreparePetDir
from nianalysis.interfaces.utils import ListDir, SelectOne, CopyToDir
from nipype.interfaces.fsl import Merge, ExtractROI
from mbianalysis.interfaces.custom.dicom import PetTimeInfo


template_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__).split('nianalysis')[0],
                 'nianalysis', 'reference_data'))


class PETStudy(Study):

#     @abstractmethod
#     def _ica_inputs(self):
#         pass

    def ICA_pipeline(self, **kwargs):
        return self._ICA_pipeline_factory(
            input_dataset=DatasetSpec('registered_volumes', nifti_gz_format))

    def _ICA_pipeline_factory(self, input_dataset, **options):

        pipeline = self.create_pipeline(
            name='ICA',
            inputs=[input_dataset],
            outputs=[DatasetSpec('decomposed_file', nifti_gz_format),
                     DatasetSpec('timeseries', nifti_gz_format),
                     DatasetSpec('mixing_mat', text_format)],
            description=('Decompose a 4D dataset into a set of independent '
                         'components using FastICA'),
            default_options={'n_components': 2, 'ica_type': 'spatial'},
            version=1,
            citations=[],
            **kwargs)

        ica = pipeline.create_node(FastICA(), name='ICA')
        ica.inputs.n_components = pipeline.option('n_components')
        ica.inputs.ica_type = pipeline.option('ica_type')
        pipeline.connect_input('registered_volumes', ica, 'volume')

        pipeline.connect_output('decomposed_file', ica, 'ica_decomposition')
        pipeline.connect_output('timeseries', ica, 'ica_timeseries')
        pipeline.connect_output('mixing_mat', ica, 'mixing_mat')

        pipeline.assert_connected()
        return pipeline

    def Image_normalization_pipeline(self, **kwargs):

        pipeline = self.create_pipeline(
            name='Image_registration',
            inputs=[DatasetSpec('pet_image', nifti_gz_format)],
            outputs=[DatasetSpec('registered_volume', nifti_gz_format),
                     DatasetSpec('warp_file', nifti_gz_format),
                     DatasetSpec('invwarp_file', nifti_gz_format),
                     DatasetSpec('affine_mat', text_matrix_format)],
            description=('Image registration to a template using ANTs'),
            default_options={'num_threads': 6, 'transformation': 's',
                             'dim': 3, 'template': (template_path +
                                                    '/PET_template.nii.gz')},
            version=1,
            citations=[],
            **kwargs)

        reg = pipeline.create_node(AntsRegSyn(out_prefix='vol2template'),
                                   name='ANTs')
        reg.inputs.num_dimensions = pipeline.option('dim')
        reg.inputs.num_threads = pipeline.option('num_threads')
        reg.inputs.transformation = pipeline.option('transformation')
        reg.inputs.ref_file = pipeline.option('template')
        pipeline.connect_input('pet_image', reg, 'input_file')

        pipeline.connect_output('registered_volume', reg, 'reg_file')
        pipeline.connect_output('warp_file', reg, 'warp_file')
        pipeline.connect_output('invwarp_file', reg, 'inv_warp')
        pipeline.connect_output('affine_mat', reg, 'regmat')
        pipeline.assert_connected()
        return pipeline

    def pet_data_preparation_pipeline(self, **kwargs):

        pipeline = self.create_pipeline(
            name='pet_data_preparation',
            inputs=[DatasetSpec('pet_recon_dir', directory_format)],
            outputs=[DatasetSpec('pet_recon_dir_prepared', directory_format)],
            description=("Given a folder with reconstructed PET data, this "
                         "pipeline will prepare the data for the motion "
                         "correction"),
            default_options={'image_orientation_check': False},
            version=1,
            citations=[],
            **kwargs)

        prep_dir = pipeline.create_node(PreparePetDir(), name='prepare_pet',
                                        requirements=[mrtrix3_req])
        prep_dir.inputs.image_orientation_check = pipeline.option(
            'image_orientation_check')
        pipeline.connect_input('pet_recon_dir', prep_dir, 'pet_dir')

        pipeline.connect_output('pet_recon_dir_prepared', prep_dir,
                                'pet_recon_dir_prepared')
        pipeline.assert_connected()
        return pipeline

    def pet_time_info_extraction_pipeline(self, **kwargs):
        pipeline = self.create_pipeline(
            name='pet_fov_cropping',
            inputs=[DatasetSpec('pet_data_dir', directory_format)],
            outputs=[FieldSpec('pet_end_time', dtype=float),
                     FieldSpec('pet_start_time', dtype=str),
                     FieldSpec('pet_duration', dtype=int)],
            description=("Extract PET time info from list-mode header."),
            version=1,
            citations=[],
            **kwargs)
    
        time_info = pipeline.create_node(PetTimeInfo(), name='PET_time_info')
        pipeline.connect_input('pet_data_dir', time_info, 'pet_data_dir')
        pipeline.connect_output('pet_end_time', time_info, 'pet_end_time')
        pipeline.connect_output('pet_start_time', time_info, 'pet_start_time')
        pipeline.connect_output('pet_duration', time_info, 'pet_duration')
        
        pipeline.assert_connected()
        return pipeline

    def pet_fov_cropping_pipeline(self, **kwargs):
        return self.pet_fov_cropping_pipeline_factory('pet2crop', **options)

    def pet_fov_cropping_pipeline_factory(self, dir2crop_name, **options):

        pipeline = self.create_pipeline(
            name='pet_fov_cropping',
            inputs=[DatasetSpec(dir2crop_name, directory_format)],
            outputs=[DatasetSpec('pet_data_cropped', directory_format)],
            description=("Given a folder with reconstructed PET data, this "
                         "pipeline will crop the PET field of view."),
            default_options={'xmin': 100, 'xsize': 130, 'ymin': 100,
                             'ysize': 130, 'zmin': 20, 'zsize': 100},
            version=1,
            citations=[],
            **kwargs)

        list_dir = pipeline.create_node(ListDir(), name='list_pet_dir')
        pipeline.connect_input('pet_recon_dir_prepared', list_dir, 'directory')
#         select = pipeline.create_node(SelectOne(), name='select_ref')
#         pipeline.connect(list_dir, 'files', select, 'inlist')
#         select.inputs.index = 0
#         crop_ref = pipeline.create_node(ExtractROI(), name='crop_ref',
#                                         requirements=[fsl509_req])
#         pipeline.connect(select, 'out', crop_ref, 'in_file')
#         crop_ref.inputs.x_min = pipeline.option('xmin')
#         crop_ref.inputs.x_size = pipeline.option('xsize')
#         crop_ref.inputs.y_min = pipeline.option('ymin')
#         crop_ref.inputs.y_size = pipeline.option('ysize')
#         crop_ref.inputs.z_min = pipeline.option('zmin')
#         crop_ref.inputs.z_size = pipeline.option('zsize')
        cropping = pipeline.create_map_node(PETFovCropping(), name='cropping',
                                            iterfield=['pet_image'])
        cropping.inputs.x_min = pipeline.option('xmin')
        cropping.inputs.x_size = pipeline.option('xsize')
        cropping.inputs.y_min = pipeline.option('ymin')
        cropping.inputs.y_size = pipeline.option('ysize')
        cropping.inputs.z_min = pipeline.option('zmin')
        cropping.inputs.z_size = pipeline.option('zsize')
#         pipeline.connect(crop_ref, 'roi_file', cropping, 'ref_pet')
        pipeline.connect(list_dir, 'files', cropping, 'pet_image')
        cp2dir = pipeline.create_node(CopyToDir(), name='copy2dir')
        pipeline.connect(cropping, 'pet_cropped', cp2dir, 'in_files')

        pipeline.connect_output('pet_data_cropped', cp2dir, 'out_dir')
        pipeline.assert_connected()
        return pipeline

    add_data_specs = [
        DatasetSpec('registered_volumes', nifti_gz_format),
        DatasetSpec('pet_image', nifti_gz_format),
        DatasetSpec('pet_data_dir', directory_format),
        DatasetSpec('pet_recon_dir', directory_format),
        DatasetSpec('pet2crop', directory_format),
        DatasetSpec('pet_recon_dir_prepared', directory_format,
                    'pet_data_preparation_pipeline'),
        DatasetSpec('pet_data_cropped', directory_format,
                    'pet_fov_cropping_pipeline'),
        DatasetSpec('decomposed_file', nifti_gz_format, 'ICA_pipeline'),
        DatasetSpec('timeseries', nifti_gz_format, 'ICA_pipeline'),
        DatasetSpec('mixing_mat', text_format, 'ICA_pipeline'),
        DatasetSpec('registered_volume', nifti_gz_format,
                    'Image_normalization_pipeline'),
        DatasetSpec('warp_file', nifti_gz_format,
                    'Image_normalization_pipeline'),
        DatasetSpec('invwarp_file', nifti_gz_format,
                    'Image_normalization_pipeline'),
        DatasetSpec('affine_mat', text_matrix_format,
                    'Image_normalization_pipeline'),
        FieldSpec('pet_duration', dtype=int,
                  pipeline=pet_time_info_extraction_pipeline),
        FieldSpec('pet_end_time', dtype=str,
                  pipeline=pet_time_info_extraction_pipeline),
        FieldSpec('pet_start_time', dtype=str,
                  pipeline=pet_time_info_extraction_pipeline)]
