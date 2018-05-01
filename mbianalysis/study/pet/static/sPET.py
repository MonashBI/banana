from ..base import PETStudy
from nianalysis.dataset import DatasetSpec
from nianalysis.study.base import StudyMetaClass
from mbianalysis.interfaces.custom.pet import SUVRCalculation
from mbianalysis.data_format import (nifti_gz_format)
import os

template_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__).split('nianalysis')[0],
                 'nianalysis', 'reference_data'))


class StaticPETStudy(PETStudy):

    __metaclass__ = StudyMetaClass

    add_data_specs = [
        DatasetSpec('pet_image', nifti_gz_format),
        DatasetSpec('base_mask', nifti_gz_format),
        DatasetSpec('SUVR_image', nifti_gz_format, 'suvr_pipeline')]

    def suvr_pipeline(self, **kwargs):

        pipeline = self.create_pipeline(
            name='SUVR',
            inputs=[DatasetSpec('registered_volume', nifti_gz_format),
                    DatasetSpec('base_mask', nifti_gz_format)],
            outputs=[DatasetSpec('SUVR_image', nifti_gz_format)],
            description=('Calculate SUVR image'),
            version=1,
            citations=[],
            **kwargs)

        suvr = pipeline.create_node(SUVRCalculation(), name='SUVR')
        pipeline.connect_input('registered_volume', suvr, 'volume')
        pipeline.connect_input('base_mask', suvr, 'base_mask')
        pipeline.connect_output('SUVR_image', suvr, 'SUVR_file')
        return pipeline

    def _ica_inputs(self):
        pass
