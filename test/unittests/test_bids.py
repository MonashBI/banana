import os
import os.path as op
import tempfile
import shutil
import logging
import wget
from zipfile import ZipFile
import subprocess as sp
from unittest import TestCase
from arcana.processor import SingleProc
from banana.bids_ import BidsRepo
from banana.utils.testing import TEST_DIR, TEST_ENV
from banana.analysis.mri import DwiAnalysis, BoldAnalysis


wf_logger = logging.getLogger('nipype.workflow')
wf_logger.setLevel(logging.INFO)
intf_logger = logging.getLogger('nipype.interface')
intf_logger.setLevel(logging.WARNING)


class TestBids(TestCase):

    BIDS_EXAMPLES_PATH = op.join(TEST_DIR, 'bids-examples-master')

    BIDS_EXAMPLES_URL = (
        'https://github.com/bids-standard/bids-examples/archive/master.zip')

    MR_DATASETS = {
        'ds000117': 17,
        'ds000246': 2,
        'ds000247': 6,
        'ds000248': 2,
        'ds001': 16,
        'ds002': 17,
        'ds003': 13,
        'ds005': 16,
        'ds006': 14,
        'ds007': 20,
        'ds008': 15,
        'ds009': 24,
        'ds011': 14,
        'ds051': 13,
        'ds052': 13,
        'ds101': 21,
        'ds102': 26,
        'ds105': 6,
        'ds107': 49,
        'ds108': 34,
        'ds109': 36,
        'ds110': 18,
        'ds113b': 20,
        'ds114': 10,
        'ds116': 17,
        'ds210': 15}

    def setUp(self):
        self.work_dir = tempfile.mkdtemp()
        # Download bids example directory if required
        if not op.exists(self.BIDS_EXAMPLES_PATH):
            tmp_zip_path = op.join(self.work_dir, 'bids-examples.zip')
            wget.download(self.BIDS_EXAMPLES_URL, tmp_zip_path)
            with ZipFile(tmp_zip_path) as zf:
                zf.extractall(path=TEST_DIR)
            os.remove(tmp_zip_path)

    def tearDown(self):
        shutil.rmtree(self.work_dir, ignore_errors=True)

    def test_project_info(self):
        for name, num_subjs in self.MR_DATASETS.items():
            print("Testing {} ({} subjects)".format(name, num_subjs))
            repo = BidsRepo(op.join(self.BIDS_EXAMPLES_PATH, name))
            self.assertEqual(len(list(repo.tree().subjects)), num_subjs)

    def test_bids_dwi(self):
        analysis = DwiAnalysis(
            'test_dwi',
            repository=BidsRepo(op.join(self.BIDS_EXAMPLES_PATH, 'ds114')),
            processor=SingleProc(self.work_dir),
            environment=TEST_ENV,
            parameters={'pe_dir': 'RL'})
        analysis.pipeline('global_tracking_pipeline')

    def test_bids_fmri(self):
        analysis = BoldAnalysis(
            'test_fmri',
            repository=BidsRepo(op.join(self.BIDS_EXAMPLES_PATH, 'ds114')),
            processor=SingleProc(self.work_dir),
            environment=TEST_ENV,
            bids_task='covertverbgeneration')
        analysis.pipeline('single_subject_melodic_pipeline')
