import os.path as op
import tempfile
import shutil
import logging
from unittest import TestCase  # @IgnorePep8
from arcana.processor import LinearProcessor, DEFAULT_PROV_IGNORE
from banana.bids import BidsRepository
from banana.utils.testing import BaseTestCase
from banana.study import DmriStudy, FmriStudy


wf_logger = logging.getLogger('nipype.workflow')
wf_logger.setLevel(logging.INFO)
intf_logger = logging.getLogger('nipype.interface')
intf_logger.setLevel(logging.WARNING)

logging.getLogger("urllib3").setLevel(logging.WARNING)


class TestBids(TestCase):

    test_dataset = op.join(BaseTestCase.test_data_dir, 'reference', 'bids',
                               'ds000114-preproc')

    def setUp(self):
        self.repo = BidsRepository(self.test_dataset)
        self.work_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.work_dir, ignore_errors=True)

    def test_project_info(self):
        tree = self.repo.tree()
        self.assertEqual(len(list(tree.subjects)), 2)
        self.assertEqual(len(list(tree.visits)), 2)

    def test_bids_dmri(self):
        study = DmriStudy(
            'test_dmri',
            repository=self.repo,
            processor=LinearProcessor(
                self.work_dir,
                prov_ignore=DEFAULT_PROV_IGNORE + [
                    'workflow/nodes/.*/requirements/.*/version'],
                reprocess=True),
            parameters={'preproc_pe_dir': 'RL'})
        study.data('tensor')

    def test_bids_fmri(self):
        study = FmriStudy(
            'test_fmri',
            repository=self.repo,
            processor=LinearProcessor(
                self.work_dir,
                prov_ignore=DEFAULT_PROV_IGNORE + [
                    'workflow/nodes/.*/requirements/.*/version']),
            parameters={'preproc_pe_dir': 'RL'})
        study.data('tensor')
