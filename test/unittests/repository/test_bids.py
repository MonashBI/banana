import os.path as op
import tempfile
import shutil
import logging
from unittest import TestCase  # @IgnorePep8
from banana.bids import BidsRepository
from banana.utils.testing import BaseTestCase
from banana.study import DmriStudy


wf_logger = logging.getLogger('nipype.workflow')
wf_logger.setLevel(logging.INFO)
intf_logger = logging.getLogger('nipype.interface')
intf_logger.setLevel(logging.WARNING)

logging.getLogger("urllib3").setLevel(logging.WARNING)


class TestBids(TestCase):

    test_dataset = op.join(BaseTestCase.test_data_dir, 'reference', 'bids',
                               'ds000114-reduced')

    def setUp(self):
        self.repo = BidsRepository(self.test_dataset)
        self.work_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.work_dir, ignore_errors=True)

    def test_project_info(self):
        tree = self.repo.tree()
        self.assertEqual(len(list(tree.subjects)), 2)
        self.assertEqual(len(list(tree.visits)), 2)

    def test_bids_selector(self):
        study = DmriStudy(
            'test_dmri',
            repository=self.repo,
            processor=self.work_dir,
            parameters={'preproc_pe_dir': 'RL'})
        study.data('tensor')
