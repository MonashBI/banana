from copy import copy
from nipype.interfaces.freesurfer.preprocess import ReconAll
# from arcana.interfaces.utils import DummyReconAll as ReconAll
from nianalysis.requirement import freesurfer_req
from nianalysis.citation import freesurfer_cites
from nianalysis.file_format import (
    freesurfer_recon_all_format, nifti_gz_format)
from arcana.dataset import DatasetSpec
from arcana.interfaces.utils import JoinPath
from ..base import MRIStudy
from arcana.study.base import StudyMetaClass
from arcana.parameter import ParameterSpec


class T1Study(MRIStudy, metaclass=StudyMetaClass):

    add_data_specs = [
        DatasetSpec('fs_recon_all', freesurfer_recon_all_format,
                    'freesurfer_pipeline'),
        DatasetSpec('brain', nifti_gz_format, 'brain_extraction_pipeline')]

    add_parameter_specs = [
        ParameterSpec('bet_method', 'optibet',
                   choices=MRIStudy.parameter_spec('bet_method').choices),
        ParameterSpec('bet_robust', True),
        ParameterSpec('bet_f_threshold', 0.57),
        ParameterSpec('bet_g_threshold', -0.1)]

    def freesurfer_pipeline(self, **kwargs):
        """
        Segments grey matter, white matter and CSF from T1 images using
        SPM "NewSegment" function.

        NB: Default values come from the W2MHS toolbox
        """
        pipeline = self.create_pipeline(
            name='segmentation',
            inputs=[DatasetSpec('primary', nifti_gz_format)],
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
        recon_all.inputs.openmp = self.processor.num_processes
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
