from nipype.pipeline import engine as pe
from nianalysis.requirements import fsl5_req
from nipype.interfaces import fsl
from nianalysis.dataset import DatasetSpec
from nianalysis.study.base import set_dataset_specs
from nianalysis.requirements import Requirement
from nianalysis.citations import fsl_cite, bet_cite, bet2_cite
from nianalysis.data_formats import nifti_gz_format
from nipype.interfaces.fsl import FNIRT
from ..base import MRStudy
from nianalysis.utils import get_atlas_path
from nianalysis.exceptions import NiAnalysisError


class StructuralStudy(MRStudy):

    def coreg_to_atlas(self, tool='fnirt', atlas='mni_nl6', **kwargs):
        if tool == 'fnirt':
            pipeline = self._fsl_fnirt_pipeline(atlas=atlas, **kwargs)
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
            inputs=self._registration_inputs,
            outputs=self._registration_outputs,
            description="Registers a MR scan against a reference image",
            options=dict(),
            requirements=[fsl5_req],
            citations=[fsl_cite],
            approx_runtime=5)
        fnirt = pe.Node(interface=FNIRT(), name='fnirt')
        atlas_image_path = get_atlas_path(atlas, 'image')
        atlas_mask_path = get_atlas_path(atlas, 'mask')
        # Set registration options
        # TODO: Need to work out which options to use
        # Connect inputs
        pipeline.connect_input('to_register', fnirt, 'in_file')
        pipeline.connect_input('reference', fnirt, 'ref_file')
        # Connect outputs
        pipeline.connect_output('registered', fnirt, 'warped_image')
        # Connect matrix
        self._connect_matrix(pipeline, fnirt)
        pipeline.assert_connected()
        return pipeline
