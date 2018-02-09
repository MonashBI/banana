from nianalysis.testing import BaseTestCase
import os.path
import logging
from nianalysis.nodes import Node
from nianalysis.interfaces.custom.qc import QAMetrics

logger = logging.getLogger('NiAnalysis')


class TestQC(BaseTestCase):

    def test_subtract(self):
        # Create Zip node
        metrics = Node(QAMetrics(), name='metrics')
        metrics.inputs.in_file = os.path.join(self.session_dir,
                                              '32ch_mprage.nii')
        out = metrics.run()
        self.assertEqual(out.outputs.snr, 1244.1515362901298)
        self.assertEqual(out.outputs.uniformity, 36.206896551724135)
        self.assertEqual(out.outputs.ghost_intensity, 29126.8961658001)
