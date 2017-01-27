from nipype.pipeline import engine as pe
from nipype.interfaces import fsl
from nianalysis.dataset import DatasetSpec
from nianalysis.study.base import Study, set_dataset_specs
from nianalysis.requirements import Requirement
from nianalysis.citations import fsl_cite, bet_cite, bet2_cite
from nianalysis.data_formats import nifti_gz_format
from nianalysis.requirements import fsl5_req
from nipype.interfaces.fsl import FNIRT, Reorient2Std
from nianalysis.utils import get_atlas_path
from nianalysis.exceptions import NiAnalysisError


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

    def coregister_to_atlas_pipeline(self, tool='fnirt', atlas='MNI152',
                                     **kwargs):
        if tool == 'fnirt':
            pipeline = self._fsl_fnirt_to_atlas_pipeline(atlas=atlas, **kwargs)
        else:
            raise NiAnalysisError("Unrecognised coregistration tool '{}'"
                                  .format(tool))
        return pipeline


    def _fsl_fnirt_to_atlas_pipeline(self, atlas, **kwargs):  # @UnusedVariable @IgnorePep8
        """
        Registers a MR scan to a refernce MR scan using FSL's nonlinear FNIRT
        command

        Parameters
        ----------
        atlas : Which atlas to use, can be one of 'mni_nl6'
        """
        pipeline = self._create_pipeline(
            name='registration',
            inputs=['primary', 'brain_mask'],
            outputs=['coreg_to_atlas', 'warp_to_atlas'],
            description="Registers a MR scan against a reference image",
            options=dict(),
            requirements=[fsl5_req],
            citations=[fsl_cite],
            approx_runtime=5)
        reorient = pe.Node(Reorient2Std(), name='reorient')
        reorient_mask = pe.Node(Reorient2Std(), name='reorient_mask')
        fnirt = pe.Node(interface=FNIRT(), name='fnirt')
        fnirt.inputs.ref_file = get_atlas_path(atlas, 'image')
        fnirt.inputs.refmask_file = get_atlas_path(atlas, 'mask')
        try:
            subsampling = kwargs['subsampling']
        except KeyError:
            subsampling = [4, 2, 1, 1]
        fnirt.inputs.subsampling_scheme = subsampling
        fnirt.inputs.field_file = True
        # Apply mask if corresponding subsampling scheme is 1
        # (i.e. 1-to-1 resolution) otherwise don't.
        fnirt.inputs.apply_inmask = [int(s == 1) for s in subsampling]
        # Connect nodes
        pipeline.connect(reorient, 'out_file', fnirt, 'in_file')
        pipeline.connect(reorient_mask, 'out_file', fnirt, 'inmask_file')
        # Set registration options
        # TODO: Need to work out which options to use
        # Connect inputs
        pipeline.connect_input('primary', reorient, 'in_file')
        pipeline.connect_input('brain_mask', reorient_mask, 'in_file')
        # Connect outputs
        pipeline.connect_output('coreg_to_atlas', fnirt, 'warped_file')
        pipeline.connect_output('warp_to_atlas', fnirt, 'field_file')
        pipeline.assert_connected()
        return pipeline

    _dataset_specs = set_dataset_specs(
        DatasetSpec('primary', nifti_gz_format),
        DatasetSpec('masked', nifti_gz_format, brain_mask_pipeline),
        DatasetSpec('brain_mask', nifti_gz_format, brain_mask_pipeline),
        DatasetSpec('coreg_to_atlas', nifti_gz_format,
                    coregister_to_atlas_pipeline),
        DatasetSpec('warp_to_atlas', nifti_gz_format,
                    coregister_to_atlas_pipeline))
