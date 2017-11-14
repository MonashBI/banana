from nianalysis.data_formats import dicom_format, nifti_format
from ..base import set_data_specs, Study
from nianalysis.dataset import DatasetSpec, FieldSpec
from nianalysis.interfaces.custom.qc import QCMetrics


class QCStudy(Study):

    def qc_metrics_pipeline(self, **options):
        pipeline = self.create_pipeline(
            name='qc_merics',
            inputs=[DatasetSpec('phantom', nifti_format)],
            outputs=[FieldSpec('snr', dtype=float),
                     FieldSpec('uniformity', dtype=float),
                     FieldSpec('ghost_intensity', dtype=float)],
            description="Registers a MR scan against a reference image",
            default_options={
                'threshold': 0.25,
                'signal_radius': 0.8,
                'ghost_radius': (1.2, 1.8),
                'background_radius': 2.25,
                'z_extent': 0.8},
            version=1,
            citations=[],
            options=options)
        metrics = pipeline.create_node(interface=QCMetrics(), name='metrics',
                                       wall_time=5)
        # Connect inputs
        pipeline.connect_input('phantom', metrics, 'in_file')
        # Connect outputs
        pipeline.connect_output('snr', metrics, 'snr')
        pipeline.connect_output('uniformity', metrics, 'uniformity')
        pipeline.connect_output('ghost_intensity', metrics, 'ghost_intensity')
        pipeline.assert_connected()
        return pipeline

    _dataset_specs = set_data_specs(
        DatasetSpec('phantom', dicom_format),
        FieldSpec('snr', dtype=float),
        FieldSpec('uniformity', dtype=float),
        FieldSpec('ghost_intensity', dtype=float))
