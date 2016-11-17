from nipype.pipeline import engine as pe
from nipype.interfaces import fsl
from nianalysis.base import Dataset
from nianalysis.study.base import _create_component_dict
from .base import MRStudy
from nianalysis.requirements import Requirement
from nianalysis.citations import fsl_cite, bet_cite, bet2_cite
from nianalysis.data_formats import nifti_gz_format


class FunctionalMRStudy(MRStudy):

    def melodic_pipeline(self, robust=True, **kwargs):  # @UnusedVariable
        """
        Generates a whole brain mask using MRtrix's 'dwi2mask' command
        """
        pipeline = self._create_pipeline(
            name='melodic',
            inputs=['mri_scan'],
            outputs=['fix', 'melodicl1'],
            description="Run FSL's MELODIC fMRI ICA analysis and FIX",
            options={},
            requirements=[Requirement('fsl', min_version=(0, 5, 0))],
            citations=[fsl_cite], approx_runtime=5)
        # Create mask node
        bet = pe.Node(interface=fsl.BET(), name="bet")
        bet.inputs.mask = True
        bet.inputs.robust = robust
        # Connect inputs/outputs
        pipeline.connect_input('mri_scan', bet, 'in_file')
        pipeline.connect_output('masked_mri_scan', bet, 'out_file')
        pipeline.connect_output('brain_mask', bet, 'mask_file')
        # Check inputs/outputs are connected
        pipeline.assert_connected()
        return pipeline

    def eroded_mask_pipeline(self, **kwargs):
        raise NotImplementedError

    _components = _create_component_dict(
        Dataset('mri_scan', nifti_gz_format),
        Dataset('masked_mri_scan', nifti_gz_format, brain_mask_pipeline),
        Dataset('brain_mask', nifti_gz_format, brain_mask_pipeline),
        Dataset('eroded_mask', nifti_gz_format, eroded_mask_pipeline))
