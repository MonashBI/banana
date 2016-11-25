from nipype.pipeline import engine as pe
from nipype.interfaces.fsl import FLIRT
from nianalysis.requirements import fsl5_req
from nianalysis.citations import fsl_cite
from nianalysis.data_formats import (
    nifti_gz_format, text_matrix_format)
from ..base import set_dataset_specs, Study
from nianalysis.dataset import DatasetSpec


class CoregisteredStudy(Study):

    _registration_inputs = ['reference', 'to_register']
    _registration_outputs = ['registered', 'matrix']

    def registration_pipeline(self, degrees_of_freedom=6,
                              cost_func='mutualinfo', qsform=False, **kwargs):  # @UnusedVariable @IgnorePep8
        """
        Registers a MR scan to a refernce MR scan using FSL's FLIRT command

        Parameters
        ----------
        degrees_of_freedom : int
            Number of degrees of freedom used in the registration. Default is
            6 -> affine transformation.
        cost_func : str
            Cost function used for the registration. Can be one of
            'mutualinfo', 'corratio', 'normcorr', 'normmi', 'leastsq',
            'labeldiff', 'bbr'
        qsform : bool
            Whether to use the QS form supplied in the input image header (
            the image coordinates of the FOV supplied by the scanner)
        """

        pipeline = self._create_pipeline(
            name='registration',
            inputs=self._registration_inputs,
            outputs=self._registration_outputs,
            description="Registers a MR scan against a reference image",
            options=dict(
                degree_of_freedom=degrees_of_freedom, cost_func=cost_func,
                qsform=qsform),
            requirements=[fsl5_req],
            citations=[fsl_cite],
            approx_runtime=5)
        flirt = pe.Node(interface=FLIRT(), name='flirt')
        # Set registration options

        flirt.inputs.dof = degrees_of_freedom
        flirt.inputs.cost = cost_func
        flirt.inputs.cost_func = cost_func
        flirt.inputs.uses_qform = qsform
        # Connect inputs
        pipeline.connect_input('to_register', flirt, 'in_file')
        pipeline.connect_input('reference', flirt, 'reference')
        # Connect outputs
        pipeline.connect_output('registered', flirt, 'out_file')
        # Connect matrix
        self._connect_matrix(pipeline, flirt)
        pipeline.assert_connected()
        return pipeline

    def _connect_matrix(self, pipeline, flirt):
        pipeline.connect_output('matrix', flirt, 'out_matrix_file')

    _dataset_specs = set_dataset_specs(
        DatasetSpec('reference', nifti_gz_format),
        DatasetSpec('to_register', nifti_gz_format),
        DatasetSpec('registered', nifti_gz_format, registration_pipeline),
        DatasetSpec('matrix', text_matrix_format, registration_pipeline))


class CoregisteredToMatrixStudy(CoregisteredStudy):
    """
    Like CoregisteredStudy but in this study the registration matrix is
    supplied as an input (typically by another sub-study)
    """

    _registration_inputs = ['reference', 'to_register', 'matrix']
    _registration_outputs = ['registered']

    def registration_pipeline(self, interpolate='trilinear', **kwargs):  # @UnusedVariable @IgnorePep8
        """
        Registers a MR scan to a refernce MR scan using FSL's FLIRT command
        using an existing registration matrix

        Parameters
        ----------
        interpolate : str
            Type of interpolation used. Can be one of 'trilinear',
            'nearestneighbour', 'sinc', 'spline', None. If None no registration
            is performed.

        (NB: see CoregisteredStudy.registration_pipeline for remaining params)
        """
        pipeline = super(
            CoregisteredToMatrixStudy, self).registration_pipeline(**kwargs)
        pipeline.node('flirt').inputs.apply_xfm = (interpolate is not None)
        if interpolate is not None:
            pipeline.node('flirt').inputs.interp = interpolate
        pipeline.options['interpolate'] = interpolate
        return pipeline

    def _connect_matrix(self, pipeline, flirt):
        pipeline.connect_input('matrix', flirt, 'in_matrix_file')

    _dataset_specs = set_dataset_specs(
        DatasetSpec('reference', nifti_gz_format),
        DatasetSpec('to_register', nifti_gz_format),
        DatasetSpec('matrix', text_matrix_format),
        DatasetSpec('registered', nifti_gz_format, registration_pipeline))
