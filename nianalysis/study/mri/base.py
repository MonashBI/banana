from nipype.pipeline import engine as pe
from nipype.interfaces import fsl
from nianalysis.dataset import DatasetSpec
from nianalysis.study.base import Study, set_dataset_specs
from nianalysis.requirements import Requirement
from nianalysis.citations import fsl_cite, bet_cite, bet2_cite
from nianalysis.data_formats import nifti_gz_format


class MRStudy(Study):

    def brain_mask_pipeline(self, robust=False, threshold=0.5,
                            reduce_bias=False, **kwargs):  # @UnusedVariable
        """
        Generates a whole brain mask using FSL's BET command
        """
        pipeline = self._create_pipeline(
            name='brain_mask',
            inputs=['primary'],
            outputs=['masked', 'brain_mask'],
            description="Generate brain mask from mr_scan",
            options=dict(robust=robust, threshold=threshold),
            requirements=[Requirement('fsl', min_version=(0, 5, 0))],
            citations=[fsl_cite, bet_cite, bet2_cite], approx_runtime=5)
        # Create mask node
        bet = pe.Node(interface=fsl.BET(), name="bet")
        bet.inputs.mask = True
        if robust:
            bet.inputs.robust = True
        if reduce_bias:
            bet.inputs.reduce_bias = True
        bet.inputs.frac = threshold
        # Connect inputs/outputs
        pipeline.connect_input('primary', bet, 'in_file')
        pipeline.connect_output('masked', bet, 'out_file')
        pipeline.connect_output('brain_mask', bet, 'mask_file')
        # Check inputs/outputs are connected
        pipeline.assert_connected()
        return pipeline

    _dataset_specs = set_dataset_specs(
        DatasetSpec('primary', nifti_gz_format),
        DatasetSpec('masked', nifti_gz_format, brain_mask_pipeline),
        DatasetSpec('brain_mask', nifti_gz_format, brain_mask_pipeline))
