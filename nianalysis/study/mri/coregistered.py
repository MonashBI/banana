from nipype.pipeline import engine as pe
from nipype.interfaces.fsl import FLIRT
from nianalysis.requirements import fsl5_req
from nianalysis.citations import fsl_cite
from nianalysis.data_formats import (
    nifti_gz_format)
from ..base import set_dataset_specs, Study
from nianalysis.dataset import DatasetSpec


class CoregisteredStudy(Study):

    def registration_pipeline(self, degree_of_freedom=6,
                              cost_func='mutualinfo',
                              search_cost_func='mutualinfo', qsform=False,
                              interpolate=None, supply_matrix=False,
                              **kwargs):  # @UnusedVariable @IgnorePep8
        """
        Registers a MR scan to a refernce MR scan using FSL's FLIRT command

        Parameters
        ----------
        degree_of_freedom : int
            Number of degrees of freedom used in the registration. Default is
            6 -> affine transformation.
        cost_func : str
            Cost function used for the registration. Can be one of
            'mutualinfo', 'corratio', 'normcorr', 'normmi', 'leastsq',
            'labeldiff', 'bbr'
        search_cost_func : str
            Cost function used for the ?? in the registration. Can be one of
            'mutualinfo', 'corratio', 'normcorr', 'normmi', 'leastsq',
            'labeldiff', 'bbr'
        qsform : bool
            Whether to use the QS form supplied in the input image header (
            the image coordinates of the FOV supplied by the scanner)
        interpolation : str
            Type of interpolation used. Can be one of 'trilinear',
            'nearestneighbour', 'sinc', 'spline', None
        supply_matrix : bool
            Supplied depending on the context (i.e. other sub-studies) in which
            the registration is performed. If the matrix is supplied by another
            registration study then it is not output here (the dataset specs
            should be mapped accordingly), otherwise it is outputted by this
            pipeline.
        """
        inputs = ['reference', 'to_register']
        outputs = ['registered']
        if supply_matrix:
            inputs.append('matrix')
        else:
            outputs.append('matrix')
        pipeline = self._create_pipeline(
            name='segmentation',
            inputs=inputs,
            outputs=outputs,
            description="Registers a MR scan against a reference image",
            options=dict(
                degree_of_freedom=degree_of_freedom, cost_func=cost_func,
                search_cost_func=search_cost_func, qsform=qsform,
                interpolate=interpolate),
            requirements=[fsl5_req],
            citations=[fsl_cite],
            approx_runtime=1)
        flirt = pe.Node(interface=FLIRT(), name='flirt')
        flirt.inputs.apply_xfm = (interpolate is not None)
        flirt.inputs.interp = interpolate
        flirt.inputs.dof = degree_of_freedom
        flirt.inputs.cost = cost_func
        flirt.inputs.cost_func = search_cost_func
        flirt.inputs.uses_qform = qsform
        # Connect inputs
        pipeline.connect_input('to_register', flirt, 'in_file')
        pipeline.connect_input('reference', flirt, 'reference')
        # Connect outputs
        pipeline.connect_output('registered', flirt, 'out_file')
        pipeline.connect_ouptut('matrix', flirt, 'out_matrix_file')
        # Connect matrix
        if supply_matrix:
            pipeline.connect_input('matrix', flirt, 'in_matrix_file')
        else:
            pipeline.connect_output('matrix', flirt, 'out_matrix_file')
        pipeline.assert_connected()
        return pipeline

    _dataset_specs = set_dataset_specs(
        DatasetSpec('reference', nifti_gz_format),
        DatasetSpec('to_register', nifti_gz_format),
        DatasetSpec('registered', nifti_gz_format, registration_pipeline),
        DatasetSpec('matrix', nifti_gz_format, registration_pipeline))
