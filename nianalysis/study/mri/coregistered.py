from nipype.pipeline import engine as pe
from nipype.interfaces.fsl import FLIRT, FNIRT
from nipype.interfaces.spm.preprocess import Coregister
from nianalysis.requirements import fsl5_req
from nianalysis.citations import fsl_cite
from nianalysis.requirements import spm12_req
from nianalysis.citations import spm_cite
from nianalysis.data_formats import (
    nifti_gz_format, nifti_format, text_matrix_format)
from ..base import set_dataset_specs, Study
from nianalysis.dataset import DatasetSpec


class CoregisteredStudy(Study):

    _registration_inputs = [DatasetSpec('reference', nifti_gz_format),
                            DatasetSpec('to_register', nifti_gz_format)]
    _registration_outputs = [DatasetSpec('registered', nifti_gz_format),
                             DatasetSpec('matrix', text_matrix_format)]

    def registration_pipeline(self, coreg_tool='flirt', **options):
        if coreg_tool == 'flirt':
            pipeline = self._fsl_flirt_pipeline(**options)
        elif coreg_tool == 'fnirt':
            pipeline = self._fsl_fnirt_pipeline(**options)
        elif coreg_tool == 'spm':
            pipeline = self._spm_coreg_pipeline(**options)
        else:
            raise NotImplementedError(
                "Unrecognised coregistration tool '{}'. Can be one of 'flirt',"
                " 'spm'.".format(coreg_tool))
        return pipeline

    def _fsl_flirt_pipeline(self, **options):  # @UnusedVariable @IgnorePep8
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

        pipeline = selfcreate_pipeline(
            name='registration_fsl',
            inputs=self._registration_inputs,
            outputs=self._registration_outputs,
            description="Registers a MR scan against a reference image",
            default_options={
                'degrees_of_freedom': 6, 'cost_func': 'mutualinfo',
                'qsform': False},
            version=1,
            requirements=[fsl5_req],
            citations=[fsl_cite],
            approx_runtime=5,
            options=options)
        flirt = pipeline.create_node(interface=FLIRT(), name='flirt')
        # Set registration options
        flirt.inputs.dof = pipeline.option('degrees_of_freedom')
        flirt.inputs.cost = pipeline.option('cost_func')
        flirt.inputs.cost_func = pipeline.option('cost_func')
        flirt.inputs.uses_qform = pipeline.option('qsform')
        # Connect inputs
        pipeline.connect_input('to_register', flirt, 'in_file')
        pipeline.connect_input('reference', flirt, 'reference')
        # Connect outputs
        pipeline.connect_output('registered', flirt, 'out_file')
        # Connect matrix
        self._connect_matrix(pipeline, flirt)
        pipeline.assert_connected()
        return pipeline

    def _fsl_fnirt_pipeline(self, **options):  # @UnusedVariable @IgnorePep8
        """
        Registers a MR scan to a refernce MR scan using FSL's nonlinear FNIRT
        command

        Parameters
        ----------
        """
        raise NotImplementedError

    def _spm_coreg_pipeline(self, **options):  # @UnusedVariable
        """
        Coregisters T2 image to T1 image using SPM's
        "Register" method.

        NB: Default values come from the W2MHS toolbox
        """
        pipeline = selfcreate_pipeline(
            name='registration_spm',
            inputs=[DatasetSpec('t1', nifti_format),
                    DatasetSpec('t2', nifti_format)],
            outputs=[DatasetSpec('t2_coreg_t1', nifti_format)],
            description="Coregister T2-weighted images to T1",
            default_options={},
            version=1,
            requirements=[spm12_req],
            citations=[spm_cite],
            approx_runtime=30,
            options=options)
        coreg = pipeline.create_node(Coregister(), name='coreg')
        coreg.inputs.jobtype = 'estwrite'
        coreg.inputs.cost_function = 'nmi'
        coreg.inputs.separation = [4, 2]
        coreg.inputs.tolerance = [
            0.02, 0.02, 0.02, 0.001, 0.001, 0.001, 0.01, 0.01, 0.01, 0.001,
            0.001, 0.001]
        coreg.inputs.fwhm = [7, 7]
        coreg.inputs.write_interp = 4
        coreg.inputs.write_wrap = [0, 0, 0]
        coreg.inputs.write_mask = False
        coreg.inputs.out_prefix = 'r'
        # Connect inputs
        pipeline.connect_input('t1', coreg, 'target')
        pipeline.connect_input('t2', coreg, 'source')
        # Connect outputs
        pipeline.connect_output('t2_coreg_t1', coreg, 'coregistered_source')
        pipeline.assert_connected()
        return pipeline

    def segmentation_pipeline(self, segment_tool='spm', **options):
        if segment_tool == 'spm':
            pipeline = self._spm_segmentation_pipeline(**options)
        else:
            raise NotImplementedError(
                "Unrecognised segmentation tool '{}'".format(segment_tool))
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

    _registration_inputs = [DatasetSpec('reference', nifti_gz_format),
                            DatasetSpec('to_register', nifti_gz_format),
                            DatasetSpec('matrix', text_matrix_format)]
    _registration_outputs = [DatasetSpec('registered', nifti_gz_format)]

    def _fsl_flirt_pipeline(self, **options):  # @UnusedVariable @IgnorePep8
        """
        Registers a MR scan to a reference MR scan with FSL's FLIRT command
        using an existing registration matrix

        Parameters
        ----------
        interpolate : str
            Type of interpolation used. Can be one of 'trilinear',
            'nearestneighbour', 'sinc', 'spline', None. If None no registration
            is performed.

        (NB: see CoregisteredStudy.registration_pipeline for remaining params)
        """
        default_interp = 'trilinear'
        pipeline = super(
            CoregisteredToMatrixStudy, self)._fsl_flirt_pipeline(**options)
        # Edit the coregister pipeline from CoregisteredStudy
        pipeline._name += '_to_matrix'
        pipeline.default_options['interpolate'] = default_interp
        pipeline.options['interpolate'] = options.get('interpolate',
                                                      default_interp)
        pipeline.create_node('flirt').inputs.apply_xfm = pipeline.option(
            'interpolate') is not None
        if pipeline.option('interpolate') is not None:
            pipeline.create_node('flirt').inputs.interp = pipeline.option(
                'interpolate')
        return pipeline

    def _spm_coreg_pipeline(self, **options):
        raise NotImplementedError(
            "SPM pipeline doesn't have (or at least it isn't implemented in "
            "NiAnalysis) a registration pipeline")

    def _connect_matrix(self, pipeline, flirt):
        pipeline.connect_input('matrix', flirt, 'in_matrix_file')

    _dataset_specs = set_dataset_specs(
        DatasetSpec('reference', nifti_gz_format),
        DatasetSpec('to_register', nifti_gz_format),
        DatasetSpec('matrix', text_matrix_format),
        DatasetSpec('registered', nifti_gz_format,
                    CoregisteredStudy.registration_pipeline))
