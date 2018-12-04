import os.path as op
from unittest import TestCase  # @IgnorePep8
from arcana.repository import Tree
from banana.repository.bids import BidsRepository
from banana.utils.testing import BaseTestCase


class TestBids(TestCase):

    test_dataset = op.join(BaseTestCase.test_data_dir, 'reference', 'bids',
                            'ds000114')

    def test_project_info(self):
        repo = BidsRepository(self.test_dataset)
        tree = repo.tree()
        self.assertEqual(len(list(tree.subjects)), 10)
        self.assertEqual(len(list(tree.visits)), 2)
