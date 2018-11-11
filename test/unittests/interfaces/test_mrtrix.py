import os.path
from arcana.utils.testing import BaseTestCase
import tempfile
from arcana.node import Node
from banana.interfaces.mrtrix import MRCalc
from banana.requirement import mrtrix_req.v('3.0')
from banana.interfaces.mrtrix.utils import ExtractFSLGradients


class TestMRCalcInterface(BaseTestCase):

    def test_subtract(self):

        tmp_dir = tempfile.mkdtemp()
        out_file = os.path.join(tmp_dir, 'out_file.mif')
        mrcalc = Node(MRCalc(), name='mrcalc',
                      requirements=[mrtrix_req.v('3.0')])
        mrcalc.inputs.operands = [os.path.join(self.session_dir,
                                               'threes.mif'),
                                  os.path.join(self.session_dir,
                                               'ones.mif')]
        mrcalc.inputs.operation = 'subtract'
        mrcalc.inputs.out_file = out_file
        mrcalc.run()
        self.assertTrue(os.path.exists(out_file))
        
    def test_extract_gradients(self):
        extract_fsl = ExtractFSLGradients()
        extract_fsl.inputs.in_file = '/Users/abha0009/Downloads/MRH060_C03_MR01/13-R_L_MRtrix_60_directions_interleaved_B0_ep2d_diff_p2'
        result = extract_fsl.run()
        self.assertTrue(os.path.exists(result.outputs.bvecs_file))
        self.assertTrue(os.path.exists(result.outputs.bvals_file))
