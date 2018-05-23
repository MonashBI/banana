import os.path
from arcana.testing import BaseTestCase
import tempfile
from arcana.node import Node
from arcana.interfaces.mrtrix import MRCalc
from nianalysis.requirement import mrtrix3_req


class TestMRCalcInterface(BaseTestCase):

    def test_subtract(self):

        tmp_dir = tempfile.mkdtemp()
        out_file = os.path.join(tmp_dir, 'out_file.mif')
        mrcalc = Node(MRCalc(), name='mrcalc',
                      requirements=[mrtrix3_req])
        mrcalc.inputs.operands = [os.path.join(self.session_dir,
                                               'threes.mif'),
                                  os.path.join(self.session_dir,
                                               'ones.mif')]
        mrcalc.inputs.operation = 'subtract'
        mrcalc.inputs.out_file = out_file
        mrcalc.run()
        self.assertTrue(os.path.exists(out_file))
