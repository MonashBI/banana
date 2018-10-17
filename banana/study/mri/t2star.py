import re
from nipype.interfaces.utility import Select
from arcana.study import StudyMetaClass
from arcana.data import FilesetSpec, AcquiredFilesetSpec
from banana.requirement import (fsl5_req, matlab2015_req,
                                    ants19_req)
from banana.citation import (
    fsl_cite, matlab_cite, sti_cites)
from banana.file_format import (
    nifti_gz_format, nifti_format, text_matrix_format,
    multi_nifti_gz_format, STD_IMAGE_FORMATS)
from banana.interfaces.custom.vein_analysis import (
    CompositeVeinImage, ShMRF)
from arcana.interfaces import utils
from .base import MriStudy
from nipype.interfaces import fsl, ants
from arcana.interfaces.utils import ListDir
from banana.interfaces.sti import (
    UnwrapPhase, VSharp, QSMiLSQR, BatchUnwrapPhase, BatchVSharp,
    BatchQSMiLSQR)
from banana.interfaces.custom.coils import HIPCombineChannels
from banana.interfaces.custom.mask import (
    DialateMask, MaskCoils, MedianInMasks)
from arcana.parameter import ParameterSpec, SwitchSpec
from banana.atlas import LocalAtlas
from logging import getLogger

logger = getLogger('banana')


def coil_sort_key(fname):
    return re.match(r'coil_(\d+)_\d+\.nii\.gz', fname).group(1)


class CoilEchoFilter():

    def __init__(self, echo):
        self._echo = echo

    def __call__(self, fname):
        match = re.match(r'coil_\d+_(\d+)', fname)
        if match is None:
            logger.warning('Ignoring file ({}) found in coil directory'
                           .format(fname))
            include = False
        else:
            include = int(match.group(1)) == self._echo
        return include


class T2StarStudy(MriStudy, metaclass=StudyMetaClass):

    add_data_specs = [
        # Set the magnitude to be generated from the preprocess_channels
        # pipeline
        FilesetSpec('magnitude', nifti_gz_format, 'preprocess_channels',
                    desc=("Generated from separate channel signals, "
                          "provided to 'channels'.")),
        # QSM and phase processing
        FilesetSpec('swi', nifti_gz_format, 'swi_pipeline'),
        FilesetSpec('qsm', nifti_gz_format, 'qsm_pipeline',
                    desc=("Quantitative susceptibility image resolved "
                                 "from T2* coil images")),
        # Vein analysis
        FilesetSpec('composite_vein_image', nifti_gz_format, 'cv_pipeline'),
        FilesetSpec('vein_mask', nifti_gz_format, 'shmrf_pipeline'),
        # Templates
        AcquiredFilesetSpec('mni_template_qsm_prior', STD_IMAGE_FORMATS,
                            frequency='per_study',
                            default=LocalAtlas('QSMPrior')),
        AcquiredFilesetSpec('mni_template_swi_prior', STD_IMAGE_FORMATS,
                            frequency='per_study',
                            default=LocalAtlas('SWIPrior')),
        AcquiredFilesetSpec('mni_template_atlas_prior', STD_IMAGE_FORMATS,
                            frequency='per_study',
                            default=LocalAtlas('VeinFrequencyPrior')),
        AcquiredFilesetSpec('mni_template_vein_atlas', STD_IMAGE_FORMATS,
                            frequency='per_study',
                            default=LocalAtlas('VeinFrequencyMap'))]

    add_param_specs = [
        SwitchSpec('qsm_dual_echo', False),
        ParameterSpec('qsm_echo', 1,
                      desc=("Which echo (by index starting at 1) to use when "
                            "using single echo")),
        ParameterSpec('qsm_padding', [12, 12, 12]),
        ParameterSpec('qsm_mask_dialation', [11, 11, 11]),
        SwitchSpec('linear_reg_method', 'ants',
                   MriStudy.parameter_spec('linear_reg_method').choices),
        SwitchSpec('bet_method', 'fsl_bet',
                   choices=MriStudy.parameter_spec('bet_method').choices),
        SwitchSpec('bet_robust', False),
        SwitchSpec('bet_robust', False),
        ParameterSpec('bet_f_threshold', 0.1),
        ParameterSpec('bet_g_threshold', 0.0)]

    def preprocess_channels(self, **name_maps):
        pipeline = super().preprocess_channels(**name_maps)
        # Connect combined first echo output to the magnitude data spec
        pipeline.connect_output('magnitude', pipeline.node('to_polar'),
                                'first_echo', nifti_gz_format)
        return pipeline

    def qsm_pipeline(self, **name_maps):
        """
        Process dual echo data for QSM (TE=[7.38, 22.14])

        NB: Default values come from the STI-Suite
        """
        pipeline = self.pipeline(
            name='qsm_pipeline',
            name_maps=name_maps,
            desc="Resolve QSM from t2star coils",
            references=[sti_cites, fsl_cite, matlab_cite])

        erosion = pipeline.add(
            'mask_erosion',
            fsl.ErodeImage(
                kernel_shape='sphere',
                kernel_size=2,
                output_type='NIFTI'),
            inputs={
                'in_file': ('brain_mask', nifti_gz_format)},
            requirements=[fsl5_req],
            wall_time=15, memory=12000)

        # If we have multiple echoes we can combine the phase images from
        # each channel into a single image. Otherwise for single echo sequences
        # we need to perform QSM on each coil separately and then combine
        # afterwards.
        if self.branch('qsm_dual_echo'):
            # Combine channels to produce phase and magnitude images
            channel_combine = pipeline.add(
                'channel_combine',
                HIPCombineChannels(),
                inputs={
                    'magnitudes_dir': ('channel_mags', multi_nifti_gz_format),
                    'phases_dir': ('channel_phases', multi_nifti_gz_format)})

            # Unwrap phase using Laplacian unwrapping
            unwrap = pipeline.add(
                'unwrap',
                UnwrapPhase(
                    padsize=self.parameter('qsm_padding')),
                inputs={
                    'voxelsize': ('voxel_sizes', float)},
                connect={
                    'in_file': (channel_combine, 'phase')})

            # Remove background noise
            vsharp = pipeline.add(
                "vsharp",
                VSharp(
                    mask_manip="imerode({}>0, ball(5))"),
                inputs={
                    'voxelsize': ('voxel_sizes', float)},
                connect={
                    'in_file': (unwrap, 'out_file'),
                    'mask': (erosion, 'out_file')})

            # Run QSM iLSQR
            pipeline.add(
                'qsmrecon',
                QSMiLSQR(
                    mask_manip="{}>0",
                    padsize=self.parameter('qsm_padding')),
                inputs={
                    'voxelsize': ('voxel_sizes', float),
                    'te': ('echo_times', float),
                    'B0': ('main_field_strength', float),
                    'H': ('main_field_orient', float)},
                connect={
                    'in_file': (vsharp, 'out_file'),
                    'mask': (vsharp, 'new_mask')},
                outputs={
                    'qsm': ('qsm', nifti_format)})

        else:
            # Dialate eroded mask
            dialate = pipeline.add(
                'dialate',
                DialateMask(
                    dialation=self.parameter('qsm_mask_dialation')),
                connect={
                    'in_file': (erosion, 'out_file')})

            # List files for the phases of separate channel
            list_phases = pipeline.add(
                'list_phases',
                ListDir(
                    sort_key=coil_sort_key,
                    filter=CoilEchoFilter(self.parameter('qsm_echo'))),
                inputs={
                    'directory': ('channel_phases', multi_nifti_gz_format)})

            # List files for the phases of separate channel
            list_mags = pipeline.add(
                'list_mags',
                ListDir(
                    sort_key=coil_sort_key,
                    filter=CoilEchoFilter(self.parameter('qsm_echo'))),
                inputs={
                    'directory': ('channel_mags', multi_nifti_gz_format)})

            # Generate coil specific masks
            mask_coils = pipeline.add(
                'mask_coils',
                MaskCoils(
                    dialation=self.parameter('qsm_mask_dialation')),
                connect={
                    'masks': (list_mags, 'files'),
                    'whole_brain_mask': (dialate, 'out_file')})

            # Unwrap phase
            unwrap = pipeline.add(
                'unwrap',
                BatchUnwrapPhase(
                    padsize=self.parameter('qsm_padding')),
                inputs={
                    'voxelsize': ('voxel_sizes', float)},
                connect={
                    'in_file': (list_phases, 'files')})

            # Background phase removal
            vsharp = pipeline.add(
                "vsharp",
                BatchVSharp(
                    mask_manip='{}>0'),
                inputs={
                    'voxelsize': ('voxel_sizes', float)},
                connect={
                    'mask': (mask_coils, 'out_files'),
                    'in_file': (unwrap, 'out_file')})

            first_echo_time = pipeline.add(
                'first_echo',
                Select(
                    index=0),
                inputs={
                    'inlist': ('echo_times', float)})

            # Perform channel-wise QSM
            coil_qsm = pipeline.add(
                'coil_qsmrecon',
                BatchQSMiLSQR(
                    mask_manip="{}>0",
                    padsize=self.parameter('qsm_padding')),
                inputs={
                    'voxelsize': ('voxel_sizes', float),
                    'B0': ('main_field_strength', float),
                    'H': ('main_field_orient', float)},
                connect={
                    'in_file': (vsharp, 'out_file'),
                    'mask': (vsharp, 'new_mask'),
                    'te': (first_echo_time, 'out')},
                wall_time=45)  # FIXME: Should be dependent on number of coils

            # Combine channel QSM by taking the median coil value
            pipeline.add(
                'combine_qsm',
                MedianInMasks(),
                connect={
                    'channels': (coil_qsm, 'out_file'),
                    'channel_masks': (vsharp, 'new_mask'),
                    'whole_brain_mask': (dialate, 'out_file')},
                outputs={
                    'out_file': ('qsm', nifti_format)})
        return pipeline

    def swi_pipeline(self, **name_maps):

        raise NotImplementedError

        pipeline = self.pipeline(
            name='swi',
            name_maps=name_maps,
            desc=("Calculate susceptibility-weighted image from magnitude and "
                  "phase"))

        return pipeline

    def cv_pipeline(self, **name_maps):

        pipeline = self.pipeline(
            name='cv_pipeline',
            name_maps=name_maps,
            desc="Compute Composite Vein Image",
            references=[fsl_cite, matlab_cite])

        # Interpolate priors and atlas
        merge_trans = pipeline.add(
            'merge_transforms',
            utils.Merge(3),
            inputs={
                'in1': ('coreg_matrix', text_matrix_format),
                'in2': ('coreg_to_atlas_mat', text_matrix_format),  # Ideally T1
                'in3': ('coreg_to_atlas_warp', nifti_gz_format)})  # Ideally T1

        apply_trans_q = pipeline.add(
            'ApplyTransform_Q_Prior',
            ants.resampling.ApplyTransforms(
                interpolation='Linear',
                input_image_type=3,
                invert_transform_flags=[True, True, False]),
            inputs={
                'input_image': ('mni_template_qsm_prior', nifti_gz_format),
                'reference_image': ('qsm', nifti_gz_format)},
            connect={
                'transforms': (merge_trans, 'out')},
            requirements=[ants19_req],
            memory=16000, wall_time=30)

        apply_trans_s = pipeline.add(
            'ApplyTransform_S_Prior',
            ants.resampling.ApplyTransforms(
                interpolation='Linear',
                input_image_type=3,
                invert_transform_flags=[True, True, False]),
            inputs={
                'input_image': ('mni_template_swi_prior', nifti_gz_format),
                'reference_image': ('qsm', nifti_gz_format)},
            connect={
                'transforms': (merge_trans, 'out')},
            requirements=[ants19_req], memory=16000, wall_time=30)

        apply_trans_a = pipeline.add(
            'ApplyTransform_A_Prior',
            ants.resampling.ApplyTransforms(
                interpolation='Linear',
                input_image_type=3,
                invert_transform_flags=[True, True, False],),
            inputs={
                'reference_image': ('qsm', nifti_gz_format),
                'input_image': ('mni_template_atlas_prior', nifti_gz_format)},
            connect={
                'transforms': (merge_trans, 'out')},
            requirements=[ants19_req],
            memory=16000, wall_time=30)

        apply_trans_v = pipeline.add(
            'ApplyTransform_V_Atlas',
            ants.resampling.ApplyTransforms(
                interpolation='Linear',
                input_image_type=3,
                invert_transform_flags=[True, True, False]),
            inputs={
                'input_image': ('mni_template_vein_atlas', nifti_gz_format),
                'reference_image': ('qsm', nifti_gz_format)},
            connect={
                'transforms': (merge_trans, 'out')},
            requirements=[ants19_req],
            memory=16000, wall_time=30)

        # Run CV code
        pipeline.add(
            'cv_image',
            interface=CompositeVeinImage(),
            inputs={
                'mask': ('brain_mask', nifti_format),
                'qsm': ('qsm', nifti_format),
                'swi': ('swi', nifti_format)},
            connect={
                'q_prior': (apply_trans_q, 'output_image'),
                's_prior': (apply_trans_s, 'output_image'),
                'a_prior': (apply_trans_a, 'output_image'),
                'vein_atlas': (apply_trans_v, 'output_image')},
            outputs={
                'out_file': ('composite_vein_image', nifti_format)},
            requirements=[matlab2015_req],
            wall_time=300, memory=24000)

        return pipeline

    def shmrf_pipeline(self, **name_maps):

        pipeline = self.pipeline(
            name='shmrf_pipeline',
            name_maps=name_maps,
            desc="Compute Vein Mask using ShMRF",
            references=[fsl_cite, matlab_cite])

        # Run ShMRF code
        pipeline.add(
            'shmrf',
            ShMRF(),
            inputs={
                'in_file': ('composite_vein_image', nifti_format),
                'mask': ('brain_mask', nifti_format)},
            outputs={
                'out_file': ('vein_mask', nifti_format)},
            requirements=[matlab2015_req],
            wall_time=30, memory=16000)

        return pipeline
