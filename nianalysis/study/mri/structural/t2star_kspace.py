from nipype.pipeline import engine as pe
from nianalysis.requirements import fsl5_req, matlab_req
from nianalysis.citations import (
    fsl_cite, matlab_cite, sti_cites)
from nianalysis.data_formats import (
    zip_format, directory_format, nifti_gz_format)
from nianalysis.study.base import set_dataset_specs
from nianalysis.dataset import DatasetSpec
from nianalysis.interfaces.qsm import STI, Prepare
from ..base import MRIStudy
from nipype.interfaces import fsl


class T2StarKSpaceStudy(MRIStudy):

    def qsm_pipeline(self, **options):  # @UnusedVariable @IgnorePep8
        """
        Segments grey matter, white matter and CSF from T1 images using
        SPM "NewSegment" function.

        NB: Default values come from the STI-Suite
        """
        pipeline = self.create_pipeline(
            name='qsmrecon',
            inputs=[DatasetSpec('t2starkspace', directory_format)],  # TODO should this be primary?
            outputs=[DatasetSpec('qsm', nifti_gz_format),
                     DatasetSpec('tissue_phase', nifti_gz_format),
                     DatasetSpec('qsm_mask', nifti_gz_format)],
            description="Resolve QSM from t2star kspace",
            default_options={},
            requirements=[fsl5_req, matlab_req],
            citations=[sti_cites, fsl_cite, matlab_cite],
            approx_runtime=30,
            version=1,
            options=options)
        # Prepare and reformat SWI_COILS
        prepare = pipeline.create_node(interface=Prepare(), name='prepare')
        # Brain Mask
        mask = pipeline.create_node(interface=fsl.BET(), name='bet')
        mask.inputs.reduce_bias = True
        mask.inputs.frac = 0.3
        mask.inputs.mask = True
        # Phase and QSM for single echo
        qsmrecon = pipeline.create_node(interface=STI(), name='qsmrecon')
        # Connect inputs/outputs
        pipeline.connect_input('t2starkspace', prepare, 'in_dir')
        pipeline.connect_output('qsm_mask', mask, 'out_file')
        pipeline.connect_output('qsm', qsmrecon, 'qsm')
        pipeline.connect_output('tissue_phase', qsmrecon, 'tissue_phase')

        pipeline.connect(prepare, 'out_file', mask, 'in_file')
        pipeline.connect(mask, 'mask_file', qsmrecon, 'mask_file')
        pipeline.connect(prepare, 'out_dir', qsmrecon, 'in_dir')

        pipeline.assert_connected()
        return pipeline

    _dataset_specs = set_dataset_specs(
        DatasetSpec('t2starkspace', zip_format),
        DatasetSpec('qsm', nifti_gz_format, qsm_pipeline),
        DatasetSpec('tissue_phase', nifti_gz_format, qsm_pipeline),
        DatasetSpec('qsm_mask', nifti_gz_format, qsm_pipeline))
