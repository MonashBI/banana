from unittest import TestCase
from nipype.interfaces.utility import IdentityInterface
from arcana.utils.testing import BaseTestCase
from banana.interfaces.mrtrix import MRConvert
from arcana.exceptions import ArcanaModulesNotInstalledException
from banana.file_format import (dicom_format, mrtrix_image_format,
                                    nifti_gz_format)
from arcana.study.base import Study, StudyMetaClass
from arcana.data import InputFilesets, FilesetSpec, InputFilesetSpec
from arcana.environment import ModulesEnv, StaticEnv

try:
    ModulesEnv._run_module_cmd('avail')
except ArcanaModulesNotInstalledException:
    environment = StaticEnv()
else:
    environment = ModulesEnv(fail_on_missing=False)


class DummyStudy(Study, metaclass=StudyMetaClass):

    add_data_specs = [
        InputFilesetSpec('input_fileset', dicom_format),
        FilesetSpec('output_fileset', nifti_gz_format, 'a_pipeline')]

    def a_pipeline(self):
        pipeline = self.new_pipeline(
            name='a_pipeline',
            inputs=[FilesetSpec('input_fileset', nifti_gz_format)],
            outputs=[FilesetSpec('output_fileset', nifti_gz_format)],
            desc=("A dummy pipeline used to test dicom-to-nifti "
                         "conversion method"),
            citations=[])
        identity = pipeline.add(
            'identity',
            IdentityInterface(['field']))
        # Connect inputs
        pipeline.connect_input('input_fileset', identity, 'field') # Connect outputs
        pipeline.connect_output('output_fileset', identity, 'field')
        return pipeline


class TestConverterAvailability(TestCase):

    def test_find_mrtrix(self):
        converter = mrtrix_image_format.converter_from(dicom_format)
        self.assertIsInstance(converter.interface, MRConvert)


class TestDicom2Niix(BaseTestCase):

    def test_dcm2niix(self):
        study = self.create_study(
            DummyStudy,
            'concatenate',
            environment=environment,
            inputs=[
                InputFilesets('input_fileset',
                                dicom_format, 't2_tse_tra_p2_448')])
        list(study.data('output_fileset'))[0]
        self.assertFilesetCreated('output_fileset.nii.gz', study.name)
