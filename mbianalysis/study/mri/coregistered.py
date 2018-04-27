from nipype.interfaces.fsl import FLIRT
from nipype.interfaces.spm.preprocess import Coregister
from nianalysis.requirements import fsl5_req
from nianalysis.citations import fsl_cite
from nianalysis.requirements import spm12_req
from nianalysis.citations import spm_cite
from nianalysis.data_formats import (
    nifti_gz_format, nifti_format, text_matrix_format)
from nianalysis.study.base import StudyMetaClass, Study
from nianalysis.dataset import DatasetSpec
from nianalysis.options import OptionSpec


class CoregisteredStudy(Study):

    __metaclass__ = StudyMetaClass

    add_data_specs = [
        DatasetSpec('reference', nifti_gz_format),
        DatasetSpec('to_register', nifti_gz_format),
        DatasetSpec('registered', nifti_gz_format,
                    'linear_registration_pipeline'),
        DatasetSpec('matrix', text_matrix_format,
                    'linear_registration_pipeline'),
        DatasetSpec('qformed', nifti_gz_format,
                    'qform_transform_pipeline'),
        DatasetSpec('qform_mat', text_matrix_format,
                    'qform_transform_pipeline')]

    add_option_specs = [
        OptionSpec('flirt_degrees_of_freedom', 6),
        OptionSpec('flirt_cost_func', 'normmi'),
        OptionSpec('flirt_qsform', False)]

    _registration_inputs = [DatasetSpec('reference', nifti_gz_format),
                            DatasetSpec('to_register', nifti_gz_format)]

    def linear_registration_pipeline(self, coreg_tool='flirt',
                                     **kwargs):
        if coreg_tool == 'flirt':
            registration_outputs = [DatasetSpec('registered', nifti_gz_format),
                                    DatasetSpec('matrix', text_matrix_format)]
            pipeline = self._fsl_flirt_pipeline(registration_outputs,
                                                **kwargs)
        elif coreg_tool == 'spm':
            pipeline = self._spm_coreg_pipeline(**kwargs)
        else:
            raise NotImplementedError(
                "Unrecognised coregistration tool '{}'. Can be one of "
                "'flirt', 'spm'.".format(coreg_tool))
        return pipeline

    def qform_transform_pipeline(self, **kwargs):

        outputs = [DatasetSpec('qformed', nifti_gz_format),
                   DatasetSpec('qform_mat', text_matrix_format)]
        reg_type = 'useqform'
        return self._fsl_flirt_pipeline(outputs, reg_type=reg_type,
                                        **kwargs)

    def _fsl_flirt_pipeline(self, outputs, reg_type='registration',
                            **kwargs):
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

        pipeline = self.create_pipeline(
            name='{}_fsl'.format(reg_type),
            inputs=self._registration_inputs,
            outputs=outputs,
            description="Registers a MR scan against a reference image",
            version=1,
            citations=[fsl_cite],
            **kwargs)
        flirt = pipeline.create_node(interface=FLIRT(), name='flirt',
                                     requirements=[fsl5_req], wall_time=5)
        if reg_type == 'useqform':
            flirt.inputs.uses_qform = True
            flirt.inputs.apply_xfm = True
            pipeline.connect_output('qformed', flirt, 'out_file')
            pipeline.connect_output('qform_mat', flirt, 'out_matrix_file')
        elif reg_type == 'registration':
            # Set registration options
            flirt.inputs.dof = pipeline.option('flirt_degrees_of_freedom')
            flirt.inputs.cost = pipeline.option('flirt_cost_func')
            flirt.inputs.cost_func = pipeline.option('flirt_cost_func')
            flirt.inputs.output_type = 'NIFTI_GZ'
            # Connect outputs
            pipeline.connect_output('registered', flirt, 'out_file')
            # Connect matrix
            self._connect_matrix(pipeline, flirt)
        # Connect inputs
        pipeline.connect_input('to_register', flirt, 'in_file')
        pipeline.connect_input('reference', flirt, 'reference')

        pipeline.assert_connected()
        return pipeline

    def _spm_coreg_pipeline(self, **kwargs):  # @UnusedVariable
        """
        Coregisters T2 image to T1 image using SPM's
        "Register" method.

        NB: Default values come from the W2MHS toolbox
        """
        pipeline = self.create_pipeline(
            name='registration_spm',
            inputs=[DatasetSpec('t1', nifti_format),
                    DatasetSpec('t2', nifti_format)],
            outputs=[DatasetSpec('t2_coreg_t1', nifti_format)],
            description="Coregister T2-weighted images to T1",
            version=1,
            citations=[spm_cite],
            **kwargs)
        coreg = pipeline.create_node(Coregister(), name='coreg',
                                     requirements=[spm12_req], wall_time=30)
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

    def _connect_matrix(self, pipeline, flirt):
        pipeline.connect_output('matrix', flirt, 'out_matrix_file')


class CoregisteredToMatrixStudy(CoregisteredStudy):
    """
    Like CoregisteredStudy but in this study the registration matrix is
    supplied as an input (typically by another sub-study)
    """

    __metaclass__ = StudyMetaClass

    _registration_inputs = [DatasetSpec('reference', nifti_gz_format),
                            DatasetSpec('to_register', nifti_gz_format),
                            DatasetSpec('matrix', text_matrix_format)]
    _registration_outputs = [DatasetSpec('registered', nifti_gz_format)]

    def _fsl_flirt_pipeline(self, outputs, **kwargs):  # @UnusedVariable @IgnorePep8
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
            CoregisteredToMatrixStudy, self)._fsl_flirt_pipeline(
                outputs, **kwargs)
        # Edit the coregister pipeline from CoregisteredStudy
        pipeline.default_options['interpolate'] = default_interp
        pipeline._options['interpolate'] = options.get('interpolate',
                                                       default_interp)
        pipeline.node('flirt').inputs.apply_xfm = pipeline.option(
            'interpolate') is not None
        if pipeline.option('interpolate') is not None:
            pipeline.node('flirt').inputs.interp = pipeline.option(
                'interpolate')
        return pipeline

    def _spm_coreg_pipeline(self, **kwargs):
        raise NotImplementedError(
            "SPM pipeline doesn't have (or at least it isn't implemented in "
            "NiAnalysis) a registration pipeline")

    def _connect_matrix(self, pipeline, flirt):
        pipeline.connect_input('matrix', flirt, 'in_matrix_file')

    add_data_specs = [
        DatasetSpec('reference', nifti_gz_format),
        DatasetSpec('to_register', nifti_gz_format),
        DatasetSpec('matrix', text_matrix_format),
        DatasetSpec('registered', nifti_gz_format,
                    CoregisteredStudy.linear_registration_pipeline)]
