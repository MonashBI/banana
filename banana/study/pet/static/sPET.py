from ..base import PETStudy
from arcana.data import FilesetSpec
from arcana.study.base import StudyMetaClass
from banana.interfaces.custom.pet import SUVRCalculation
from banana.file_format import (nifti_gz_format)
import os

template_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__).split('arcana')[0],
                 'arcana', 'reference_data'))


class StaticPETStudy(PETStudy, metaclass=StudyMetaClass):

    add_data_specs = [
        FilesetSpec('pet_image', nifti_gz_format),
        FilesetSpec('base_mask', nifti_gz_format),
        FilesetSpec('SUVR_image', nifti_gz_format, 'suvr_pipeline')]

    def suvr_pipeline(self, **kwargs):

        pipeline = self.pipeline(
            name='SUVR',
            inputs=[FilesetSpec('registered_volume', nifti_gz_format),
                    FilesetSpec('base_mask', nifti_gz_format)],
            outputs=[FilesetSpec('SUVR_image', nifti_gz_format)],
            desc=('Calculate SUVR image'),
            references=[],
            **kwargs)

        suvr = pipeline.create_node(SUVRCalculation(), name='SUVR')
        pipeline.connect_input('registered_volume', suvr, 'volume')
        pipeline.connect_input('base_mask', suvr, 'base_mask')
        pipeline.connect_output('SUVR_image', suvr, 'SUVR_file')
        return pipeline

    def _ica_inputs(self):
        pass
