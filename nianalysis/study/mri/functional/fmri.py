from nipype.pipeline import engine as pe
from nipype.interfaces import fsl
from nianalysis.dataset import DatasetSpec
from nianalysis.study.base import set_dataset_specs
from ..base import MRStudy
from nianalysis.requirements import Requirement
from nianalysis.citations import fsl_cite
from nianalysis.data_formats import nifti_gz_format


class FunctionalMRStudy(MRStudy):

    def melodic_pipeline(self, **options):  # @UnusedVariable
        """
        Generates a whole brain mask using MRtrix's 'dwi2mask' command
        """
        pipeline = self._create_pipeline(
            name='melodic',
            inputs=['mri_scan'],
            outputs=['fix', 'melodicl1'],
            description="Run FSL's MELODIC fMRI ICA analysis and FIX",
            default_options={'robust': True},
            requirements=[Requirement('fsl', min_version=(0, 5, 0))],
            citations=[fsl_cite], approx_runtime=5, options=options)
        # Create mask node
        bet = pe.Node(interface=SomeInterface(), name="fmri")
        bet.inputs.mask = True
        bet.inputs.robust = self.options['robust']
        # Connect inputs/outputs
        pipeline.connect_input('mri_scan', bet, 'in_file')
        pipeline.connect_output('masked_mri_scan', bet, 'out_file')
        pipeline.connect_output('brain_mask', bet, 'mask_file')
        # Check inputs/outputs are connected
        pipeline.assert_connected()
        return pipeline

    def eroded_mask_pipeline(self, **options):
        raise NotImplementedError

    _dataset_specs = set_dataset_specs(
        DatasetSpec('primary', nifti_gz_format))
