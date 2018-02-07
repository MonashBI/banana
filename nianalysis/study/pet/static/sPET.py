from ..base import PETStudy
from nianalysis.dataset import DatasetSpec
from nianalysis.study.base import set_dataset_specs
from nianalysis.interfaces.custom import SUVRCalculation
from nianalysis.data_formats import (nifti_gz_format)
import os

template_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__).split('nianalysis')[0],
                 'nianalysis', 'reference_data'))


class StaticPETStudy(PETStudy):

    def suvr_pipeline(self, **options):

        pipeline = self.create_pipeline(
            name='SUVR',
            inputs=[DatasetSpec('registered_volume', nifti_gz_format),
                    DatasetSpec('base_mask', nifti_gz_format)],
            outputs=[DatasetSpec('SUVR_image', nifti_gz_format)],
            description=('Calculate SUVR image'),
            default_options={},
            version=1,
            citations=[],
            options=options)

        suvr = pipeline.create_node(SUVRCalculation(), name='SUVR')
        pipeline.connect_input('registered_volume', suvr, 'volume')
        pipeline.connect_input('base_mask', suvr, 'base_mask')
        pipeline.connect_output('SUVR_image', suvr, 'SUVR_file')
        pipeline.assert_connected()
        return pipeline

    def _ica_inputs(self):
        pass

    _dataset_specs = set_dataset_specs(
        DatasetSpec('pet_image', nifti_gz_format),
        DatasetSpec('base_mask', nifti_gz_format),
        DatasetSpec('SUVR_image', nifti_gz_format, suvr_pipeline),
        inherit_from=PETStudy.generated_dataset_specs())
