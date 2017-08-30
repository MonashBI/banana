from nianalysis.study.base import Study, set_dataset_specs
from nianalysis.dataset import DatasetSpec
from nianalysis.data_formats import (nifti_gz_format, text_format,
                                     text_matrix_format)
from nianalysis.interfaces.sklearn import FastICA
from nianalysis.interfaces.ants import AntsRegSyn
import os


template_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__).split('nianalysis')[0],
                 'nianalysis', 'reference_data'))


class PETStudy(Study):

    def ICA_pipeline(self, **options):

        pipeline = self.create_pipeline(
            name='ICA',
            inputs=[DatasetSpec('registered_volumes', nifti_gz_format)],
            outputs=[DatasetSpec('decomposed_file', nifti_gz_format),
                     DatasetSpec('timeseries', nifti_gz_format),
                     DatasetSpec('mixing_mat', text_format)],
            description=('Decompose a 4D dataset into a set of independent '
                         'components using FastICA'),
            default_options={'n_components': 2, 'ica_type': 'spatial'},
            version=1,
            citations=[],
            options=options)

        ica = pipeline.create_node(FastICA(), name='ICA')
        ica.inputs.n_components = pipeline.option('n_components')
        ica.inputs.ica_type = pipeline.option('ica_type')
        pipeline.connect_input('registered_volumes', ica, 'volume')

        pipeline.connect_output('decomposed_file', ica, 'ica_decomposition')
        pipeline.connect_output('timeseries', ica, 'ica_timeseries')
        pipeline.connect_output('mixing_mat', ica, 'mixing_mat')

        pipeline.assert_connected()
        return pipeline

    def Image_normalization_pipeline(self, **options):

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
            options=options)

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

    _dataset_specs = set_dataset_specs(
        DatasetSpec('registered_volumes', nifti_gz_format),
        DatasetSpec('pet_image', nifti_gz_format),
        DatasetSpec('decomposed_file', nifti_gz_format, ICA_pipeline),
        DatasetSpec('timeseries', nifti_gz_format, ICA_pipeline),
        DatasetSpec('mixing_mat', text_format, ICA_pipeline),
        DatasetSpec('registered_volume', nifti_gz_format,
                    Image_normalization_pipeline),
        DatasetSpec('warp_file', nifti_gz_format,
                    Image_normalization_pipeline),
        DatasetSpec('invwarp_file', nifti_gz_format,
                    Image_normalization_pipeline),
        DatasetSpec('affine_mat', text_matrix_format,
                    Image_normalization_pipeline))
