import os.path
import shutil
from unittest import TestCase
from nipype.pipeline import engine as pe
from nianalysis.base import Scan
from nianalysis.formats import nifti_gz_format
from nianalysis.requirements import mrtrix3_req
from nianalysis.dataset.base import Dataset, _create_component_dict
from nianalysis.interfaces.mrtrix import MRConvert
from nianalysis.archive.local import LocalArchive


class DummyDataset(Dataset):

    def pipeline1(self):
        pipeline = self._create_pipeline(
            name='pipeline1',
            inputs=['start'],
            outputs=['pipeline1_1', 'pipeline1_2'],
            description="A dummy pipeline used to test 'run_pipeline' method",
            options={},
            requirements=[mrtrix3_req],
            citations=[],
            approx_runtime=1)
        mrconvert = pe.Node(MRConvert(), name="convert1")
        mrconvert2 = pe.Node(MRConvert(), name="convert2")
        # Connect inputs
        pipeline.connect_input('start', mrconvert, 'in_file')
        pipeline.connect_input('start', mrconvert2, 'in_file')
        # Connect outputs
        pipeline.connect_output('pipeline1_1', mrconvert, 'out_file')
        pipeline.connect_output('pipeline1_2', mrconvert2, 'out_file')
        # Check inputs/outputs are connected
        pipeline.assert_connected()
        return pipeline

    def pipeline2(self):
        pipeline = self._create_pipeline(
            name='pipeline2',
            inputs=['start', 'pipeline1_1'],
            outputs=['pipeline2'],
            description="A dummy pipeline used to test 'run_pipeline' method",
            options={},
            requirements=[mrtrix3_req],
            citations=[],
            approx_runtime=1)
        mrmath = pe.Node(MRMath(), name="math")
        # Connect inputs
        pipeline.connect_input('start', mrmath, 'in_file')
        # Connect outputs
        pipeline.connect_output('pipeline2', mrmath, 'out_file')
        # Check inputs/outputs are connected
        pipeline.assert_connected()
        return pipeline

    def pipeline3(self):
        pipeline = self._create_pipeline(
            name='pipeline3',
            inputs=['pipeline1_2'],
            outputs=['pipeline3'],
            description="A dummy pipeline used to test 'run_pipeline' method",
            options={},
            requirements=[mrtrix3_req],
            citations=[],
            approx_runtime=1)
        mrconvert = pe.Node(MRConvert(), name="convert")
        # Connect inputs
        pipeline.connect_input('pipeline1_2', mrconvert, 'in_file')
        # Connect outputs
        pipeline.connect_output('pipeline3', mrconvert, 'out_file')
        # Check inputs/outputs are connected
        pipeline.assert_connected()
        return pipeline

    def pipeline4(self):
        pipeline = self._create_pipeline(
            name='pipeline4',
            inputs=['pipeline3'],
            outputs=['pipeline4'],
            description="A dummy pipeline used to test 'run_pipeline' method",
            options={},
            requirements=[mrtrix3_req],
            citations=[],
            approx_runtime=1)
        mrconvert = pe.Node(MRConvert(), name="convert")
        # Connect inputs
        pipeline.connect_input('pipeline3', mrconvert, 'in_file')
        # Connect outputs
        pipeline.connect_output('pipeline4', mrconvert, 'out_file')
        # Check inputs/outputs are connected
        pipeline.assert_connected()
        return pipeline

    _components = _create_component_dict(
        Scan('start', nifti_gz_format),
        Scan('pipeline1_1', nifti_gz_format, pipeline1),
        Scan('pipeline1_2', nifti_gz_format, pipeline1),
        Scan('pipeline2', nifti_gz_format, pipeline2),
        Scan('pipeline3', nifti_gz_format, pipeline3),
        Scan('pipeline4', nifti_gz_format, pipeline4))


class TestRunPipeline(TestCase):

    PROJECT_ID = 'DUMMYPROJECTID'
    SUBJECT_ID = 'DUMMYSUBJECTID'
    SESSION_ID = 'DUMMYSESSIONID'
    TEST_IMAGE = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', '..', '_data', 'test_image.nii.gz'))
    TEST_DIR = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', '..', '_data', 'dataset'))
    BASE_DIR = os.path.abspath(os.path.join(TEST_DIR, 'base_dir'))
    WORKFLOW_DIR = os.path.abspath(os.path.join(TEST_DIR, 'workflow_dir'))

    def setUp(self):
        # Create test data on DaRIS
        self._study_id = None
        # Make cache and working dirs
        shutil.rmtree(self.TEST_DIR, ignore_errors=True)
        os.makedirs(self.WORKFLOW_DIR)
        self.session_path = os.path.join(
            self.BASE_DIR, self.PROJECT_ID, self.SUBJECT_ID, self.SESSION_ID)
        os.makedirs(self.session_path)
        shutil.copy(self.TEST_IMAGE,
                    os.path.join(self.session_path, 'start.nii.gz'))

    def tearDown(self):
        # Clean up working dirs
        shutil.rmtree(self.TEST_DIR, ignore_errors=True)

    def test_run_pipeline(self):
        archive = LocalArchive(self.BASE_DIR)
        dataset = DummyDataset(
            'TestDummy', self.PROJECT_ID, archive,
            input_scans={'start': Scan('start', nifti_gz_format)})
        dataset.pipeline4().run()
        for scan in DummyDataset.components:
            self.assertTrue(
                os.path.exists(os.path.join(
                    self.session_path, scan.name + scan.format.extension)))