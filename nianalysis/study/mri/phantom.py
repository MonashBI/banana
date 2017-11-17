from nianalysis.data_formats import dicom_format, nifti_format, nifti_gz_format
from ..base import set_data_specs, Study
from nianalysis.dataset import DatasetSpec, FieldSpec
from nianalysis.interfaces.custom.qc import QCMetrics


class PhantomStudy(Study):

    def qc_metrics_pipeline(self, **options):
        pipeline = self.create_pipeline(
            name='qc_merics',
            inputs=[DatasetSpec('phantom', nifti_format)],
            outputs=[FieldSpec('snr', dtype=float),
                     FieldSpec('uniformity', dtype=float),
                     FieldSpec('ghost_intensity', dtype=float),
                     DatasetSpec('signal', format=nifti_format),
                     DatasetSpec('ghost', format=nifti_format),
                     DatasetSpec('background', format=nifti_format)],
            description="Registers a MR scan against a reference image",
            default_options={
                'threshold': 0.25,
                'signal_radius': 0.8,
                'ghost_radii': (1.1, 1.5),
                'background_radius': 1.6,
                'z_extent': 0.8},
            version=1,
            citations=[],
            options=options)
        metrics = pipeline.create_node(interface=QCMetrics(), name='metrics',
                                       wall_time=5)
        metrics.inputs.threshold = pipeline.option('threshold')
        metrics.inputs.signal_radius = pipeline.option('signal_radius')
        metrics.inputs.ghost_radii = pipeline.option('ghost_radii')
        metrics.inputs.background_radius = pipeline.option('background_radius')
        metrics.inputs.z_extent = pipeline.option('z_extent')
        # Connect inputs
        pipeline.connect_input('phantom', metrics, 'in_file')
        # Connect outputs
        pipeline.connect_output('snr', metrics, 'snr')
        pipeline.connect_output('uniformity', metrics, 'uniformity')
        pipeline.connect_output('ghost_intensity', metrics, 'ghost_intensity')
        pipeline.connect_output('signal', metrics, 'signal')
        pipeline.connect_output('ghost', metrics, 'ghost')
        pipeline.connect_output('background', metrics, 'background')
        pipeline.assert_connected()
        return pipeline

    _data_specs = set_data_specs(
        DatasetSpec('phantom', dicom_format),
        DatasetSpec('signal', format=nifti_gz_format,
                    pipeline=qc_metrics_pipeline),
        DatasetSpec('ghost', format=nifti_gz_format,
                    pipeline=qc_metrics_pipeline),
        DatasetSpec('background', format=nifti_gz_format,
                    pipeline=qc_metrics_pipeline),
        FieldSpec('snr', dtype=float, pipeline=qc_metrics_pipeline),
        FieldSpec('uniformity', dtype=float, pipeline=qc_metrics_pipeline),
        FieldSpec('ghost_intensity', dtype=float,
                  pipeline=qc_metrics_pipeline))
