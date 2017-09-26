from nianalysis.testing import BaseTestCase
import os.path
import logging
from nianalysis.nodes import Node
from nianalysis.interfaces.custom.qc import QCMetrics

logger = logging.getLogger('NiAnalysis')


class TestQC(BaseTestCase):

    def test_subtract(self):
        # Create Zip node
        metrics = Node(QCMetrics(), name='metrics')
        metrics.inputs.in_file = os.path.join(self.session_dir,
                                              '32ch_mprage.nii')
        out = metrics.run()
        self.assertEqual(out.outputs.snr, 23.838683913521724)
        self.assertEqual(out.outputs.uniformity, 23.838683913521724)
        self.assertEqual(out.outputs.ghost_intensity, 23.838683913521724)
