from unittest import TestCase
from nipype.interfaces.utility import IdentityInterface
from arcana.testing import BaseTestCase
from arcana.interfaces.mrtrix import MRConvert
from arcana.exception import ArcanaModulesNotInstalledException
from arcana.data.file_format import (
    Converter)
from nianalysis.file_format import (dicom_format, mrtrix_format,
                                    nifti_gz_format)
from arcana.requirement import Requirement
from arcana.node import Node
from arcana.study.base import Study, StudyMetaClass
from arcana.data import FilesetSelector, FilesetSpec


dummy_req = Requirement('name-for-module-that-will-never-exist',
                        min_version=(0, 3, 12))


class DummyConverter(Converter):

    requirements = [dummy_req]

    def get_node(self, node_name, input_format, output_format):  # @UnusedVariable @IgnorePep8
        convert_node = Node(IdentityInterface(['in_out']), name=node_name,
                            requirements=self.requirements)
        return convert_node, 'in_out', 'in_out'


class DummyStudy(Study, metaclass=StudyMetaClass):

    add_data_specs = [
        FilesetSpec('input_fileset', dicom_format),
        FilesetSpec('output_fileset', nifti_gz_format, 'a_pipeline')]

    def a_pipeline(self):
        pipeline = self.pipeline(
            name='a_pipeline',
            inputs=[FilesetSpec('input_fileset', nifti_gz_format)],
            outputs=[FilesetSpec('output_fileset', nifti_gz_format)],
            desc=("A dummy pipeline used to test dicom-to-nifti "
                         "conversion method"),
            references=[])
        identity = pipeline.create_node(IdentityInterface(['field']),
                                        name='identity')
        # Connect inputs
        pipeline.connect_input('input_fileset', identity, 'field')
        # Connect outputs
        pipeline.connect_output('output_fileset', identity, 'field')
        return pipeline


class TestConverterAvailability(TestCase):

    def setUp(self):
        try:
            Node.available_modules()
            self.modules_installed = True
        except ArcanaModulesNotInstalledException:
            self.modules_installed = False

    def test_find_mrtrix(self):
        if self.modules_installed:
            converter = mrtrix_format.converter_from(dicom_format)
            node, _, _ = converter.get_node('dummy')
            self.assertIsInstance(node.interface, MRConvert)


class TestDicom2Niix(BaseTestCase):

    def test_dcm2niix(self):
        study = self.create_study(
            DummyStudy, 'concatenate', inputs=[
                FilesetSelector('input_fileset',
                             dicom_format, 't2_tse_tra_p2_448')])
        list(study.data('output_fileset'))[0]
        self.assertFilesetCreated('output_fileset.nii.gz', study.name)
