from nianalysis.study.base import Study, set_dataset_specs
from nianalysis.dataset import DatasetSpec
from nianalysis.data_formats import nifti_gz_format, text_format
from nianalysis.interfaces.sklearn import FastICA


class PETStudy(Study):

    def ICA_pipeline(self, **options):

        pipeline = self.create_pipeline(
            name='Independent Component Analysis (ICA)',
            inputs=[DatasetSpec('volume', nifti_gz_format)],
            outputs=[DatasetSpec('decomposed_file', nifti_gz_format),
                     DatasetSpec('timeseries', nifti_gz_format),
                     DatasetSpec('mixing_mat', text_format)],
            description=('Decompose a 4D dataset into a set of independent '
                         'components using FastICA'),
            default_options={'n_components': 2, 'ica_type': 'spatial'},
            version=1,
            options=options)

        ica = pipeline.create_node(FastICA, name='ICA')
        ica.inputs.n_components = pipeline.option('n_components')
        ica.inputs.ica_type = pipeline.option('ica_type')
        pipeline.connect_input('volume', ica, 'volume')

        pipeline.assert_connected()
        return pipeline

    _dataset_specs = set_dataset_specs(
        DatasetSpec('volume', nifti_gz_format),
        DatasetSpec('decomposed_file', nifti_gz_format, ICA_pipeline),
        DatasetSpec('timeseries', nifti_gz_format, ICA_pipeline),
        DatasetSpec('mixing_mat', text_format, ICA_pipeline))
