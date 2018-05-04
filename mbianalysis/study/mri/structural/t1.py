from copy import copy
from nipype.interfaces.freesurfer.preprocess import ReconAll
# from nianalysis.interfaces.utils import DummyReconAll as ReconAll
from mbianalysis.requirement import freesurfer_req
from mbianalysis.citation import freesurfer_cites
from mbianalysis.data_format import (
    freesurfer_recon_all_format, mrconvert_nifti_gz_format)
from nianalysis.dataset import DatasetSpec
from nianalysis.interfaces.utils import JoinPath
from ..base import MRIStudy
from nianalysis.study.base import StudyMetaClass
from nianalysis.option import OptionSpec


class T1Study(MRIStudy):

    __metaclass__ = StudyMetaClass

    add_data_specs = [
        DatasetSpec('fs_recon_all', freesurfer_recon_all_format,
                    'freesurfer_pipeline'),
        DatasetSpec('brain', mrconvert_nifti_gz_format, 'brain_mask_pipeline')]

    add_option_specs = [
        OptionSpec('bet_method', 'optibet',
                   choices=MRIStudy.option_spec('bet_method').choices),
        OptionSpec('bet_robust', True),
        OptionSpec('bet_f_threshold', 0.57),
        OptionSpec('bet_g_threshold', -0.1)]

    def freesurfer_pipeline(self, **kwargs):
        """
        Segments grey matter, white matter and CSF from T1 images using
        SPM "NewSegment" function.

        NB: Default values come from the W2MHS toolbox
        """
        pipeline = self.create_pipeline(
            name='segmentation',
            inputs=[DatasetSpec('primary', mrconvert_nifti_gz_format)],
            outputs=[DatasetSpec('fs_recon_all',
                                 freesurfer_recon_all_format)],
            desc="Segment white/grey matter and csf",
            version=1,
            citations=copy(freesurfer_cites),
            **kwargs)
        # FS ReconAll node
        recon_all = pipeline.create_node(
            interface=ReconAll(), name='recon_all',
            requirements=[freesurfer_req], wall_time=2000)
        recon_all.inputs.directive = 'all'
        recon_all.inputs.openmp = self.runner.num_processes
        # Wrapper around os.path.join
        join = pipeline.create_node(interface=JoinPath(), name='join')
        pipeline.connect(recon_all, 'subjects_dir', join, 'dirname')
        pipeline.connect(recon_all, 'subject_id', join, 'filename')
        # Connect inputs/outputs
        pipeline.connect_input('primary', recon_all, 'T1_files')
        pipeline.connect_output('fs_recon_all', join, 'path')
        return pipeline

    def segmentation_pipeline(self, **kwargs):
        pipeline = super(T1Study, self).segmentation_pipeline(img_type=1,
                                                              **kwargs)
        return pipeline
