from itertools import chain
from copy import copy
from nipype.interfaces.freesurfer.preprocess import ReconAll
# from nianalysis.interfaces.utils import DummyReconAll as ReconAll
from nianalysis.requirements import freesurfer_req
from nianalysis.citations import (
    freesurfer_cites, optimal_t1_bet_params_cite)
from nianalysis.data_formats import (freesurfer_recon_all_format,
                                     nifti_gz_format)
from nianalysis.study.base import set_data_specs
from nianalysis.dataset import DatasetSpec
from nianalysis.interfaces.utils import ZipDir, JoinPath
from ..base import MRIStudy


class T1Study(MRIStudy):

    def brain_mask_pipeline(self, robust=False, threshold=0.1,
                            reduce_bias=True, **kwargs):
        pipeline = super(T1Study, self).brain_mask_pipeline(
            robust=robust, threshold=threshold, reduce_bias=reduce_bias,
            **kwargs)
        pipeline.citations.append(optimal_t1_bet_params_cite)

    def freesurfer_pipeline(self, num_processes=16, **options):  # @UnusedVariable @IgnorePep8
        """
        Segments grey matter, white matter and CSF from T1 images using
        SPM "NewSegment" function.

        NB: Default values come from the W2MHS toolbox
        """
        pipeline = self.create_pipeline(
            name='segmentation',
            inputs=[DatasetSpec('primary', nifti_gz_format)],
            outputs=[DatasetSpec('fs_recon_all', freesurfer_recon_all_format)],
            description="Segment white/grey matter and csf",
            default_options={},
            version=1,
            citations=copy(freesurfer_cites),
            options=options)
        # FS ReconAll node
        recon_all = pipeline.create_node(
            interface=ReconAll(), name='recon_all',
            requirements=[freesurfer_req], wall_time=2000)
        recon_all.inputs.directive = 'all'
        recon_all.inputs.openmp = num_processes
        # Wrapper around os.path.join
        join = pipeline.create_node(interface=JoinPath(), name='join')
        pipeline.connect(recon_all, 'subjects_dir', join, 'dirname')
        pipeline.connect(recon_all, 'subject_id', join, 'filename')
        # Zip directory before returning
        zip_dir = pipeline.create_node(interface=ZipDir(), name='zip_dir')
        zip_dir.inputs.extension = '.fs'
        pipeline.connect(join, 'path', zip_dir, 'dirname')
        # Connect inputs/outputs
        pipeline.connect_input('primary', recon_all, 'T1_files')
        pipeline.connect_output('fs_recon_all', zip_dir, 'zipped')
        pipeline.assert_connected()
        return pipeline

    _data_specs = set_data_specs(
        DatasetSpec('fs_recon_all', freesurfer_recon_all_format,
                    freesurfer_pipeline),
        inherit_from=chain(MRIStudy.data_specs()))
