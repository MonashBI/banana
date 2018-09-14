import os.path
from arcana.testing import BaseTestCase
from nipype.pipeline.engine import Node
from nianalysis.interfaces.custom.ct import Ct2Umap


class TestCT2Umap(BaseTestCase):

    def test_registration(self):
        ct2umap = Node(Ct2Umap(), name='ct2umap')
        ct2umap.inputs.ct_reg = '/Users/apoz0003/git/copy/ct.nii.gz'
        ct2umap.inputs.a = 0.0000564
        ct2umap.inputs.b = 0.0408
        ct2umap.inputs.BP =1030
        results = ct2umap.run()
        results.outputs.sute_fix_template
        self.assertTrue(os.path.exists(results.outputs.sute_fix_template))
