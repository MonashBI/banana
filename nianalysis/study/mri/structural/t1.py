from copy import copy
from nipype.interfaces.freesurfer.preprocess import ReconAll
# from arcana.interfaces.utils import DummyReconAll as ReconAll
from nianalysis.requirement import freesurfer_req, ants19_req, fsl5_req
from nianalysis.citation import freesurfer_cites, fsl_cite
from nipype.interfaces import fsl, ants
from arcana.interfaces import utils
from nianalysis.file_format import (
    freesurfer_recon_all_format, nifti_gz_format, text_matrix_format)
from arcana.data import FilesetSpec
from arcana.interfaces.utils import JoinPath
from ..base import MRIStudy
from arcana.study.base import StudyMetaClass
from arcana.parameter import ParameterSpec, SwitchSpec


class T1Study(MRIStudy, metaclass=StudyMetaClass):

    add_data_specs = [
        FilesetSpec('fs_recon_all', freesurfer_recon_all_format,
                    'freesurfer_pipeline'),
        FilesetSpec('brain', nifti_gz_format, 'brain_extraction_pipeline')]

    add_parameter_specs = [
        SwitchSpec('bet_method', 'optibet',
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
        pipeline = self.new_pipeline(
            name='segmentation',
            inputs=[FilesetSpec('magnitude', nifti_gz_format)],
            outputs=[FilesetSpec('fs_recon_all',
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
        pipeline.connect_input('magnitude', recon_all, 'T1_files')
        pipeline.connect_output('fs_recon_all', join, 'path')
        return pipeline

    def segmentation_pipeline(self, **kwargs):
        pipeline = super(T1Study, self).segmentation_pipeline(img_type=1,
                                                              **kwargs)
        return pipeline

    def bet_T1(self, **options):

        pipeline = self.create_pipeline(
            name='BET_T1',
            inputs=[FilesetSpec('t1', nifti_gz_format)],
            outputs=[FilesetSpec('betted_T1', nifti_gz_format),
                     FilesetSpec('betted_T1_mask', nifti_gz_format)],
            description=("python implementation of BET"),
            default_options={},
            version=1,
            citations=[fsl_cite],
            options=options)

        bias = pipeline.create_node(interface=ants.N4BiasFieldCorrection(),
                                    name='n4_bias_correction',
                                    requirements=[ants19_req],
                                    wall_time=60, memory=12000)
        pipeline.connect_input('t1', bias, 'input_image')

        bet = pipeline.create_node(
            fsl.BET(frac=0.15, reduce_bias=True), name='bet',
            requirements=[fsl5_req], memory=8000, wall_time=45)

        pipeline.connect(bias, 'output_image', bet, 'in_file')
        pipeline.connect_output('betted_T1', bet, 'out_file')
        pipeline.connect_output('betted_T1_mask', bet, 'mask_file')

        return pipeline

    def cet_T1(self, **options):
        pipeline = self.create_pipeline(
            name='CET_T1',
            inputs=[FilesetSpec('betted_T1', nifti_gz_format),
                    FilesetSpec(
                self._lookup_l_tfm_to_name('MNI'),
                text_matrix_format),
                FilesetSpec(self._lookup_nl_tfm_inv_name('MNI'),
                            nifti_gz_format)],
            outputs=[FilesetSpec('cetted_T1_mask', nifti_gz_format),
                     FilesetSpec('cetted_T1', nifti_gz_format)],
            description=("Construct cerebellum mask using SUIT template"),
            default_options={
                'SUIT_mask': self._lookup_template_mask_path('SUIT')},
            version=1,
            citations=[fsl_cite],
            options=options)

        # Initially use MNI space to warp SUIT into T1 and threshold to mask
        merge_trans = pipeline.create_node(
            utils.Merge(2), name='merge_transforms')
        pipeline.connect_input(
            self._lookup_nl_tfm_inv_name('MNI'),
            merge_trans,
            'in2')
        pipeline.connect_input(
            self._lookup_l_tfm_to_name('MNI'),
            merge_trans,
            'in1')

        apply_trans = pipeline.create_node(
            ants.resampling.ApplyTransforms(), name='ApplyTransform',
            requirements=[ants19_req], memory=16000, wall_time=120)
        apply_trans.inputs.interpolation = 'NearestNeighbor'
        apply_trans.inputs.input_image_type = 3
        apply_trans.inputs.invert_transform_flags = [True, False]
        apply_trans.inputs.input_image = pipeline.option('SUIT_mask')

        pipeline.connect(merge_trans, 'out', apply_trans, 'transforms')
        pipeline.connect_input('betted_T1', apply_trans, 'reference_image')

        maths2 = pipeline.create_node(
            fsl.utils.ImageMaths(
                suffix='_optiBET_cerebellum',
                op_string='-mas'),
            name='mask', requirements=[fsl5_req], memory=16000, wall_time=5)
        pipeline.connect_input('betted_T1', maths2, 'in_file')
        pipeline.connect(apply_trans, 'output_image', maths2, 'in_file2')

        pipeline.connect_output('cetted_T1', maths2, 'out_file')
        pipeline.connect_output('cetted_T1_mask', apply_trans, 'output_image')

        return pipeline
