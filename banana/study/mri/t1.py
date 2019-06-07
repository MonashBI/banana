from copy import copy
import itertools
from nipype.interfaces.freesurfer.preprocess import ReconAll
# from arcana.utils.interfaces import DummyReconAll as ReconAll
from nipype.interfaces import fsl, ants, mrtrix3
from arcana.utils.interfaces import Merge
from arcana.data import FilesetSpec, InputFilesetSpec
from arcana.utils.interfaces import JoinPath
from arcana.study.base import StudyMetaClass
from arcana.study import ParamSpec, SwitchSpec
from arcana.utils.interfaces import CopyToDir
from banana.requirement import freesurfer_req, ants_req, fsl_req, mrtrix_req
from banana.citation import freesurfer_cites, fsl_cite
from banana.interfaces.freesurfer import AparcStats
from banana.interfaces.utility import AppendPath
from banana.file_format import (
    nifti_gz_format, nifti_format, zip_format, STD_IMAGE_FORMATS,
    directory_format, text_format, mrtrix_image_format)
from banana.reference import LocalReferenceData
from .t2 import T2Study


class T1Study(T2Study, metaclass=StudyMetaClass):

    add_data_specs = [
        FilesetSpec('fs_recon_all', zip_format, 'freesurfer_pipeline'),
        InputFilesetSpec(
            't2_coreg', STD_IMAGE_FORMATS, optional=True,
            desc=("A coregistered T2 image to use in freesurfer to help "
                  "distinguish the peel surface")),
        # Templates
        InputFilesetSpec('suit_mask', STD_IMAGE_FORMATS,
                         frequency='per_study',
                         default=LocalReferenceData('SUIT', nifti_format)),
        FilesetSpec('five_tissue_type', mrtrix_image_format,
                    'gen_5tt_pipeline',
                    desc=("A segmentation image taken from freesurfer output "
                          "and simplified into 5 tissue types. Used in ACT "
                          "streamlines tractography"))] + [
        FilesetSpec(
            'aparc_stats_{}_{}_table'.format(h, m),
            text_format, 'aparc_stats_table_pipeline',
            frequency='per_visit',
            pipeline_args={'hemisphere': h, 'measure': m},
            desc=("Table of {} of {} per parcellated segment"
                  .format(m, h.upper())))
        for h, m in itertools.product(
            ('lh', 'rh'),
            ('volume', 'thickness', 'thicknessstd', 'meancurv', 'gauscurv',
             'foldind', 'curvind'))]

    add_param_specs = [
        # MriStudy.param_spec('bet_method').with_new_choices(default='opti_bet'),
        SwitchSpec('bet_robust', False),
        SwitchSpec('bet_reduce_bias', True),
        SwitchSpec('aparc_atlas', 'desikan-killiany',
                   choices=('desikan-killiany', 'destrieux', 'DKT')),
        ParamSpec('bet_f_threshold', 0.1),
        ParamSpec('bet_g_threshold', 0.0)]

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
                'T1_files': ('mag_preproc', nifti_gz_format)},
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

    def gen_5tt_pipeline(self, **name_maps):

        pipeline = self.new_pipeline(
            name='gen5tt',
            name_maps=name_maps,
            desc=("Generate 5-tissue-type image used for Anatomically-"
                  "Constrained Tractography (ACT)"))

        aseg_path = pipeline.add(
            'aseg_path',
            AppendPath(
                sub_paths=['mri', 'aseg.mgz']),
            inputs={
                'base_path': ('fs_recon_all', directory_format)})

        pipeline.add(
            'gen5tt',
            mrtrix3.Generate5tt(
                algorithm='freesurfer',
                out_file='5tt.mif'),
            inputs={
                'in_file': (aseg_path, 'out_path')},
            outputs={
                'five_tissue_type': ('out_file', mrtrix_image_format)},
            requirements=[mrtrix_req.v('3.0rc3'),
                          freesurfer_req.v('6.0')])

        return pipeline

    def aparc_stats_table_pipeline(self, measure, hemisphere, **name_maps):

        pipeline = self.new_pipeline(
            name='aparc_stats_{}_{}'.format(hemisphere, measure),
            name_maps=name_maps,
            desc=("Extract statistics from freesurfer outputs"))

        copy_to_dir = pipeline.add(
            'copy_to_subjects_dir',
            CopyToDir(),
            inputs={
                'in_files': ('fs_recon_all', directory_format),
                'file_names': (self.SUBJECT_ID, int)},
            joinsource=self.SUBJECT_ID,
            joinfield=['in_files', 'file_names'])

        if self.branch('aparc_atlas', 'desikan-killiany'):
            parc = 'aparc'
        elif self.branch('aparc_atlas', 'destrieux'):
            parc = 'aparc.a2009s'
        elif self.branch('aparc_atlas', 'DKT'):
            parc = 'aparc.DKTatlas40'
        else:
            self.unhandled_branch('aparc_atlas')

        pipeline.add(
            'aparc_stats',
            AparcStats(
                measure=measure,
                hemisphere=hemisphere,
                parc=parc),
            inputs={
                'subjects_dir': (copy_to_dir, 'out_dir'),
                'subjects': (copy_to_dir, 'file_names')},
            outputs={
                'aparc_stats_{}_{}_table'
                .format(hemisphere, measure): ('tablefile', text_format)},
            requirements=[freesurfer_req.v('5.3')])

        return pipeline

    def bet_T1(self, **name_maps):

        pipeline = self.new_pipeline(
            name='BET_T1',
            name_maps=name_maps,
            desc=("Brain extraction pipeline using FSL's BET"),
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
            fsl.BET(
                frac=0.15,
                reduce_bias=True,
                output_type='NIFTI_GZ'),
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


class PreclinicalT1(T1Study, metaclass=StudyMetaClass):

    pass
