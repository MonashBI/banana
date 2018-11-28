import os
from arcana.utils.testing import BaseTestCase
import tempfile
from nipype.pipeline import Node
from banana.interfaces.custom.coils import CombineCoils


class TestMRCalcInterface(BaseTestCase):

    def test_subtract(self):

        tmp_dir = tempfile.mkdtemp()
        orig_dir = os.getcwd()
        try:
            os.chdir(tmp_dir)
            combine = Node(CombineCoils(), name='combine')
            combine.inputs.in_dir = '/Users/tclose/Downloads/swi_coils'
            result = combine.run()
            out_files = os.listdir(result.outputs.coils_dir)
            self.assertEqual(out_files, [])
        finally:
            os.chdir(orig_dir)

