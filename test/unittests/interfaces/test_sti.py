import os.path as op
import unittest
from banana.utils.testing import BaseTestCase, TEST_ENV
import tempfile
from arcana.environment.base import Node
from banana.interfaces.sti import UnwrapPhase
from banana.requirement import sti_req, matlab_req


class TestMRCalcInterface(BaseTestCase):

    @unittest.skip
    def test_unwrap(self):

        tmp_dir = tempfile.mkdtemp()
        in_file = op.join(tmp_dir, 'in_file.nii.gz')
        with open(in_file, 'w') as f:
            f.write('test')
        out_file = op.join(tmp_dir, 'out_file.nii.gz')
        unwrap = TEST_ENV.make_node(UnwrapPhase(), name='unwrap',
                                    requirements=[matlab_req.v('r2018a'),
                                                  sti_req.v(2.2)])
        unwrap.inputs.in_file = in_file
        unwrap.inputs.voxelsize = [2.0, 2.0, 2.0]
#         unwrap.inputs.out_file = out_file
        unwrap.run()
        self.assertTrue(op.exists(out_file))
