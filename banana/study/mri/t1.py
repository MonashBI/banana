from copy import copy
from nipype.interfaces.freesurfer.preprocess import ReconAll
# from arcana.utils.interfaces import DummyReconAll as ReconAll
from banana.requirement import freesurfer_req, ants_req, fsl_req
from banana.citation import freesurfer_cites, fsl_cite
from nipype.interfaces import fsl, ants
from arcana.utils.interfaces import Merge
from banana.file_format import (
    nifti_gz_format, nifti_format, zip_format, STD_IMAGE_FORMATS,
    directory_format)
from arcana.data import FilesetSpec, FilesetInputSpec
from arcana.utils.interfaces import JoinPath
from .t2 import T2Study, MriStudy
from arcana.study.base import StudyMetaClass
from arcana.study import ParamSpec, SwitchSpec
from banana.reference import LocalReferenceData


class T1Study(T2Study, metaclass=StudyMetaClass):

    add_data_specs = [
        FilesetSpec('fs_recon_all', zip_format, 'freesurfer_pipeline'),
        FilesetSpec('brain', nifti_gz_format, 'brain_extraction_pipeline'),
        FilesetInputSpec(
            't2_coreg', STD_IMAGE_FORMATS, optional=True,
            desc=("A coregistered T2 image to use in freesurfer to help "
                  "distinguish the peel surface")),
        # Templates
        FilesetInputSpec('suit_mask', STD_IMAGE_FORMATS,
                         frequency='per_study',
                         default=LocalReferenceData('SUIT', nifti_format))]

    add_param_specs = [
        SwitchSpec('bet_method', 'fsl_bet',
                   choices=MriStudy.parameter_spec('bet_method').choices),
        SwitchSpec('bet_robust', False),
        SwitchSpec('bet_reduce_bias', True),
        ParamSpec('bet_f_threshold', 0.1),
        ParamSpec('bet_g_threshold', 0.0)]
#         SwitchSpec('bet_method', 'optibet',
#                    choices=MriStudy.parameter_spec('bet_method').choices),
#         SwitchSpec('bet_robust', True),
#         ParamSpec('bet_f_threshold', 0.57),
#         ParamSpec('bet_g_threshold', -0.1)]

    def freesurfer_pipeline(self, **name_maps):
        """
        Segments grey matter, white matter and CSF from T1 images using
        SPM "NewSegment" function.

        NB: Default values come from the W2MHS toolbox
        """
        pipeline = self.new_pipeline(
            name='segmentation',
            name_maps=name_maps,
            desc="Segment white/grey matter and csf",
            citations=copy(freesurfer_cites))

        # FS ReconAll node
        recon_all = pipeline.add(
            'recon_all',
            interface=ReconAll(
                directive='all',
                openmp=self.processor.num_processes),
            inputs={
                'T1_files': ('preproc', nifti_gz_format)},
            requirements=[freesurfer_req.v('5.3')],
            wall_time=2000)

        if self.provided('t2_coreg'):
            pipeline.connect_input('t2_coreg', recon_all, 'T2_file',
                                   nifti_gz_format)
            recon_all.inputs.use_T2 = True

        # Wrapper around os.path.join
        pipeline.add(
            'join',
            JoinPath(),
            inputs={
                'dirname': (recon_all, 'subjects_dir'),
                'filename': (recon_all, 'subject_id')},
            outputs={
                'fs_recon_all': ('path', directory_format)})

        return pipeline

    def segmentation_pipeline(self, **name_maps):
        pipeline = super(T1Study, self).segmentation_pipeline(img_type=1,
                                                              **name_maps)
        return pipeline

    def bet_T1(self, **name_maps):

        pipeline = self.new_pipeline(
            name='BET_T1',
            name_maps=name_maps,
            desc=("python implementation of BET"),
            citations=[fsl_cite])

        bias = pipeline.add(
            'n4_bias_correction',
            ants.N4BiasFieldCorrection(),
            inputs={
                'input_image': ('t1', nifti_gz_format)},
            requirements=[ants_req.v('1.9')],
            wall_time=60, mem_gb=12)

        pipeline.add(
            'bet',
            fsl.BET(frac=0.15, reduce_bias=True),
            inputs={
                'in_file': (bias, 'output_image')},
            outputs={
                'betted_T1': ('out_file', nifti_gz_format),
                'betted_T1_mask': ('mask_file', nifti_gz_format)},
            requirements=[fsl_req.v('5.0.8')], mem_gb=8,
            wall_time=45)

        return pipeline

    def cet_T1(self, **name_maps):
        pipeline = self.new_pipeline(
            name='CET_T1',
            name_maps=name_maps,
            desc=("Construct cerebellum mask using SUIT template"),
            citations=[fsl_cite])

        # FIXME: Should convert to inputs
        nl = self._lookup_nl_tfm_inv_name('MNI')
        linear = self._lookup_l_tfm_to_name('MNI')

        # Initially use MNI space to warp SUIT into T1 and threshold to mask
        merge_trans = pipeline.add(
            'merge_transforms',
            Merge(2),
            inputs={
                'in2': (nl, nifti_gz_format),
                'in1': (linear, nifti_gz_format)})

        apply_trans = pipeline.add(
            'ApplyTransform',
            ants.resampling.ApplyTransforms(
                interpolation='NearestNeighbor',
                input_image_type=3,
                invert_transform_flags=[True, False]),
            inputs={
                'reference_image': ('betted_T1', nifti_gz_format),
                'input_image': ('suit_mask', nifti_gz_format),
                'transforms': (merge_trans, 'out')},
            requirements=[ants_req.v('1.9')], mem_gb=16,
            wall_time=120)

        pipeline.add(
            'maths2',
            fsl.utils.ImageMaths(
                suffix='_optiBET_cerebellum',
                op_string='-mas'),
            inputs={
                'in_file': ('betted_T1', nifti_gz_format),
                'in_file2': (apply_trans, 'output_image')},
            outputs={
                'cetted_T1': ('out_file', nifti_gz_format),
                'cetted_T1_mask': ('output_image', nifti_gz_format)},
            requirements=[fsl_req.v('5.0.8')], mem_gb=16,
            wall_time=5)

        return pipeline
