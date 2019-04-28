from .base import PetStudy
from arcana.data import FilesetSpec
from arcana.study.base import StudyMetaClass
from banana.interfaces.custom.pet import SUVRCalculation
from banana.file_format import (nifti_gz_format)
import os

template_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__).split('arcana')[0],
                 'arcana', 'reference'))


class StaticPetStudy(PetStudy, metaclass=StudyMetaClass):

    add_data_specs = [
        FilesetSpec('pet_image', nifti_gz_format),
        FilesetSpec('base_mask', nifti_gz_format),
        FilesetSpec('SUVR_image', nifti_gz_format, 'suvr_pipeline')]

    def suvr_pipeline(self, **kwargs):

        pipeline = self.new_pipeline(
            name='SUVR',
            desc=('Calculate SUVR image'),
            citations=[],
            **kwargs)

        pipeline.add(
            'SUVR',
            SUVRCalculation(),
            inputs={
                'volume': ('registered_volume', nifti_gz_format),
                'base_mask': ('base_mask', nifti_gz_format)},
            outputs={
                'SUVR_image': ('SUVR_file', nifti_gz_format)})

        return pipeline

    def _ica_inputs(self):
        pass
