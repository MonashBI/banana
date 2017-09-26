from nianalysis.data_formats import dicom_format, nifti_gz_format
from ..base import set_dataset_specs, Study
from nianalysis.dataset import DatasetSpec


class QCStudy(Study):

    def qc_analysis_pipeline(self, **options):
        pipeline = self.create_pipeline(
            name='qc_',
            inputs=self._registration_inputs,
            outputs=self._registration_outputs,
            description="Registers a MR scan against a reference image",
            default_options={
                'degrees_of_freedom': 6, 'cost_func': 'mutualinfo',
                'qsform': False},
            version=1,
            citations=[],
            options=options)
        flirt = pipeline.create_node(interface=FLIRT(), name='flirt',
                                     requirements=[fsl5_req], wall_time=5)
        # Set registration options
        flirt.inputs.dof = pipeline.option('degrees_of_freedom')
        flirt.inputs.cost = pipeline.option('cost_func')
        flirt.inputs.cost_func = pipeline.option('cost_func')
        flirt.inputs.uses_qform = pipeline.option('qsform')
        flirt.inputs.output_type = 'NIFTI_GZ'
        # Connect inputs
        pipeline.connect_input('to_register', flirt, 'in_file')
        pipeline.connect_input('reference', flirt, 'reference')
        # Connect outputs
        pipeline.connect_output('registered', flirt, 'out_file')
        # Connect matrix
        self._connect_matrix(pipeline, flirt)
        pipeline.assert_connected()
        return pipeline

    _dataset_specs = set_dataset_specs(
        DatasetSpec('qc', dicom_format),
        DatasetSpec('to_register', nifti_gz_format))
