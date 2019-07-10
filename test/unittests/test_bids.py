import os.path as op
import tempfile
import shutil
import logging
from unittest import TestCase
from arcana.processor import SingleProc
from banana.bids_ import BidsRepo
from banana.utils.testing import TEST_DIR
from banana.study.mri import DwiStudy, BoldStudy


wf_logger = logging.getLogger('nipype.workflow')
wf_logger.setLevel(logging.INFO)
intf_logger = logging.getLogger('nipype.interface')
intf_logger.setLevel(logging.WARNING)

logging.getLogger("urllib3").setLevel(logging.WARNING)


class TestBids(TestCase):

    test_dataset = op.join(TEST_DIR, 'reference', 'bids', 'ds000114-preproc')

    def setUp(self):
        self.repo = BidsRepo(self.test_dataset)
        self.work_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.work_dir, ignore_errors=True)

    def test_project_info(self):
        tree = self.repo.tree()
        self.assertEqual(len(list(tree.subjects)), 2)
        self.assertEqual(len(list(tree.visits)), 2)

    def test_bids_dwi(self):
        study = DwiStudy(
            'test_dwi',
            repository=self.repo,
            processor=SingleProc(
                self.work_dir,
                prov_ignore=SingleProc.DEFAULT_PROV_IGNORE + [
                    'workflow/nodes/.*/requirements/.*/version'],
                reprocess=True),
            parameters={'preproc_pe_dir': 'RL'})
        study.data('tensor')

    def test_bids_fmri(self):
        study = BoldStudy(
            'test_fmri',
            repository=self.repo,
            processor=SingleProc(
                self.work_dir,
                prov_ignore=SingleProc.DEFAULT_PROV_IGNORE + [
                    'workflow/nodes/.*/requirements/.*/version']),
            bids_task='covertverbgeneration')
        study.data('melodic_ica')
