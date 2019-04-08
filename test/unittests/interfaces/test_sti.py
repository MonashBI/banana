import os.path as op
from arcana.utils.testing import BaseTestCase
import tempfile
from nipype.pipeline.engine import Node
from banana.interfaces.sti import UnwrapPhase
# from banana.requirement import sti_req


class TestMRCalcInterface(BaseTestCase):

    def test_subtract(self):

        tmp_dir = tempfile.mkdtemp()
        in_file = op.join(tmp_dir, 'in_file.nii.gz')
        with open(in_file, 'w') as f:
            f.write('test')
        out_file = op.join(tmp_dir, 'out_file.nii.gz')
        unwrap = Node(UnwrapPhase(), name='unwrap')
        unwrap.inputs.in_file = in_file
        unwrap.inputs.voxelsize = [2.0, 2.0, 2.0]
#         unwrap.inputs.out_file = out_file
        unwrap.run()
        self.assertTrue(op.exists(out_file))
