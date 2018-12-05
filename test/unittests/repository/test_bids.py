import os.path as op
import tempfile
import shutil
from unittest import TestCase  # @IgnorePep8
from banana.bids import BidsSelector, BidsRepository
from banana.utils.testing import BaseTestCase
from banana.study import DmriStudy


class TestBids(TestCase):

    test_dataset = op.join(BaseTestCase.test_data_dir, 'reference', 'bids',
                               'ds000114')

    def setUp(self):
        self.repo = BidsRepository(self.test_dataset)
        self.work_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.work_dir, ignore_errors=True)

    def test_project_info(self):
        tree = self.repo.tree()
        self.assertEqual(len(list(tree.subjects)), 10)
        self.assertEqual(len(list(tree.visits)), 2)

    def test_bids_selector(self):
        tree = self.repo.tree()
        study = DmriStudy(
            'test_dmri',
            repository=self.repo,
            processor=self.work_dir,
            inputs=[BidsSelector(name='magnitude', type='dwi')])
        study.data('magnitude')

