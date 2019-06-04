import re
from nipype.interfaces.utility import Select
from arcana.study import StudyMetaClass
from arcana.data import FilesetSpec, InputFilesetSpec
from banana.requirement import (fsl_req, matlab_req, ants_req, sti_req)
from banana.citation import (
    fsl_cite, matlab_cite, sti_cites)
from banana.file_format import (
    nifti_gz_format, nifti_format, text_matrix_format,
    multi_nifti_gz_format, STD_IMAGE_FORMATS)
from banana.interfaces.custom.vein_analysis import (
    CompositeVeinImage, ShMRF)
from arcana.utils.interfaces import Merge
from .base import MriStudy
from nipype.interfaces import fsl, ants
from arcana.utils import get_class_info
from arcana.utils.interfaces import ListDir
from banana.interfaces.sti import (
    UnwrapPhase, VSharp, QSMiLSQR, BatchUnwrapPhase, BatchVSharp,
    BatchQSMiLSQR)
from banana.interfaces.custom.coils import HIPCombineChannels
from banana.interfaces.custom.mask import (
    DialateMask, MaskCoils, MedianInMasks)
from arcana.study import ParamSpec, SwitchSpec
from banana.reference import LocalReferenceData
from logging import getLogger

logger = getLogger('banana')


class CoilSortKey():

    def __call__(self, fname):
        return re.match(r'coil_(\d+)_\d+\.nii\.gz', fname).group(1)

    @property
    def prov(self):
        return {'type': get_class_info(type(self))}


coil_sort_key = CoilSortKey()


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

    @property
    def prov(self):
        return {'type': get_class_info(type(self)),
                'echo': self._echo}


class T2starStudy(MriStudy, metaclass=StudyMetaClass):

    add_data_specs = [
        # Set the magnitude to be generated from the preprocess_channels
        # pipeline
        FilesetSpec('magnitude', nifti_gz_format,
                    'preprocess_channels_pipeline',
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
        InputFilesetSpec('mni_template_qsm_prior', STD_IMAGE_FORMATS,
                            frequency='per_study',
                            default=LocalReferenceData('QSMPrior',
                                                       nifti_gz_format)),
        InputFilesetSpec('mni_template_swi_prior', STD_IMAGE_FORMATS,
                            frequency='per_study',
                            default=LocalReferenceData('SWIPrior',
                                                       nifti_gz_format)),
        InputFilesetSpec('mni_template_atlas_prior', STD_IMAGE_FORMATS,
                            frequency='per_study',
                            default=LocalReferenceData('VeinFrequencyPrior',
                                                       nifti_gz_format)),
        InputFilesetSpec('mni_template_vein_atlas', STD_IMAGE_FORMATS,
                            frequency='per_study',
                            default=LocalReferenceData('VeinFrequencyMap',
                                                       nifti_gz_format))]

    add_param_specs = [
        SwitchSpec('qsm_dual_echo', False),
        ParamSpec('qsm_echo', 1,
                      desc=("Which echo (by index starting at 1) to use when "
                            "using single echo")),
        ParamSpec('qsm_padding', [12, 12, 12]),
        ParamSpec('qsm_mask_dialation', [11, 11, 11]),
        ParamSpec('qsm_erosion_size', 10),
        SwitchSpec('bet_robust', False),
        SwitchSpec('bet_robust', False),
        ParamSpec('bet_f_threshold', 0.1),
        ParamSpec('bet_g_threshold', 0.0)]

    def preprocess_channels_pipeline(self, **name_maps):
        pipeline = super().preprocess_channels_pipeline(**name_maps)
        # Connect combined first echo output to the magnitude data spec
        pipeline.connect_output('magnitude', pipeline.node('to_polar'),
                                'first_echo', nifti_gz_format)
        return pipeline

    def qsm_pipeline(self, **name_maps):
        """
        Process dual echo data for QSM (TE=[7.38, 22.14])

        NB: Default values come from the STI-Suite
        """
        pipeline = self.new_pipeline(
            name='qsm_pipeline',
            name_maps=name_maps,
            desc="Resolve QSM from t2star coils",
            citations=[sti_cites, fsl_cite, matlab_cite])

        erosion = pipeline.add(
            'mask_erosion',
            fsl.ErodeImage(
                kernel_shape='sphere',
                kernel_size=self.parameter('qsm_erosion_size'),
                output_type='NIFTI'),
            inputs={
                'in_file': ('brain_mask', nifti_gz_format)},
            requirements=[fsl_req.v('5.0.8')],
            wall_time=15, mem_gb=12)

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
                    'magnitudes_dir': ('mag_channels', multi_nifti_gz_format),
                    'phases_dir': ('phase_channels', multi_nifti_gz_format)})

            # Unwrap phase using Laplacian unwrapping
            unwrap = pipeline.add(
                'unwrap',
                UnwrapPhase(
                    padsize=self.parameter('qsm_padding')),
                inputs={
                    'voxelsize': ('voxel_sizes', float),
                    'in_file': (channel_combine, 'phase')},
                requirements=[matlab_req.v('r2017a'), sti_req.v(2.2)])

            # Remove background noise
            vsharp = pipeline.add(
                "vsharp",
                VSharp(
                    mask_manip="imerode({}>0, ball(5))"),
                inputs={
                    'voxelsize': ('voxel_sizes', float),
                    'in_file': (unwrap, 'out_file'),
                    'mask': (erosion, 'out_file')},
                requirements=[matlab_req.v('r2017a'), sti_req.v(2.2)])

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
                    'H': ('main_field_orient', float),
                    'in_file': (vsharp, 'out_file'),
                    'mask': (vsharp, 'new_mask')},
                outputs={
                    'qsm': ('qsm', nifti_format)},
                requirements=[matlab_req.v('r2017a'), sti_req.v(2.2)])

        else:
            # Dialate eroded mask
            dialate = pipeline.add(
                'dialate',
                DialateMask(
                    dialation=self.parameter('qsm_mask_dialation')),
                inputs={
                    'in_file': (erosion, 'out_file')},
                requirements=[matlab_req.v('r2017a')])

            # List files for the phases of separate channel
            list_phases = pipeline.add(
                'list_phases',
                ListDir(
                    sort_key=coil_sort_key,
                    filter=CoilEchoFilter(self.parameter('qsm_echo'))),
                inputs={
                    'directory': ('phase_channels', multi_nifti_gz_format)})

            # List files for the phases of separate channel
            list_mags = pipeline.add(
                'list_mags',
                ListDir(
                    sort_key=coil_sort_key,
                    filter=CoilEchoFilter(self.parameter('qsm_echo'))),
                inputs={
                    'directory': ('mag_channels', multi_nifti_gz_format)})

            # Generate coil specific masks
            mask_coils = pipeline.add(
                'mask_coils',
                MaskCoils(
                    dialation=self.parameter('qsm_mask_dialation')),
                inputs={
                    'masks': (list_mags, 'files'),
                    'whole_brain_mask': (dialate, 'out_file')},
                requirements=[matlab_req.v('r2017a')])

            # Unwrap phase
            unwrap = pipeline.add(
                'unwrap',
                BatchUnwrapPhase(
                    padsize=self.parameter('qsm_padding')),
                inputs={
                    'voxelsize': ('voxel_sizes', float),
                    'in_file': (list_phases, 'files')},
                requirements=[matlab_req.v('r2017a'), sti_req.v(2.2)])

            # Background phase removal
            vsharp = pipeline.add(
                "vsharp",
                BatchVSharp(
                    mask_manip='{}>0'),
                inputs={
                    'voxelsize': ('voxel_sizes', float),
                    'mask': (mask_coils, 'out_files'),
                    'in_file': (unwrap, 'out_file')},
                requirements=[matlab_req.v('r2017a'), sti_req.v(2.2)])

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
                    'H': ('main_field_orient', float),
                    'in_file': (vsharp, 'out_file'),
                    'mask': (vsharp, 'new_mask'),
                    'te': (first_echo_time, 'out')},
                requirements=[matlab_req.v('r2017a'), sti_req.v(2.2)],
                wall_time=45)  # FIXME: Should be dependent on number of coils

            # Combine channel QSM by taking the median coil value
            pipeline.add(
                'combine_qsm',
                MedianInMasks(),
                inputs={
                    'channels': (coil_qsm, 'out_file'),
                    'channel_masks': (vsharp, 'new_mask'),
                    'whole_brain_mask': (dialate, 'out_file')},
                outputs={
                    'qsm': ('out_file', nifti_format)},
                requirements=[matlab_req.v('r2017a')])
        return pipeline

    def swi_pipeline(self, **name_maps):

        raise NotImplementedError

        pipeline = self.new_pipeline(
            name='swi',
            name_maps=name_maps,
            desc=("Calculate susceptibility-weighted image from magnitude and "
                  "phase"))

        return pipeline

    def cv_pipeline(self, **name_maps):

        pipeline = self.new_pipeline(
            name='cv_pipeline',
            name_maps=name_maps,
            desc="Compute Composite Vein Image",
            citations=[fsl_cite, matlab_cite])

        # Interpolate priors and atlas
        merge_trans = pipeline.add(
            'merge_transforms',
            Merge(3),
            inputs={
                'in1': ('coreg_ants_mat', text_matrix_format),
                'in2': ('coreg_to_tmpl_ants_mat', text_matrix_format),
                'in3': ('coreg_to_tmpl_ants_warp', nifti_gz_format)})

        apply_trans_q = pipeline.add(
            'ApplyTransform_Q_Prior',
            ants.resampling.ApplyTransforms(
                interpolation='Linear',
                input_image_type=3,
                invert_transform_flags=[True, True, False]),
            inputs={
                'input_image': ('mni_template_qsm_prior', nifti_gz_format),
                'reference_image': ('qsm', nifti_gz_format),
                'transforms': (merge_trans, 'out')},
            requirements=[ants_req.v('1.9')],
            mem_gb=16,
            wall_time=30)

        apply_trans_s = pipeline.add(
            'ApplyTransform_S_Prior',
            ants.resampling.ApplyTransforms(
                interpolation='Linear',
                input_image_type=3,
                invert_transform_flags=[True, True, False]),
            inputs={
                'input_image': ('mni_template_swi_prior', nifti_gz_format),
                'reference_image': ('qsm', nifti_gz_format),
                'transforms': (merge_trans, 'out')},
            requirements=[ants_req.v('1.9')], mem_gb=16,
            wall_time=30)

        apply_trans_a = pipeline.add(
            'ApplyTransform_A_Prior',
            ants.resampling.ApplyTransforms(
                interpolation='Linear',
                input_image_type=3,
                invert_transform_flags=[True, True, False],),
            inputs={
                'reference_image': ('qsm', nifti_gz_format),
                'input_image': ('mni_template_atlas_prior', nifti_gz_format),
                'transforms': (merge_trans, 'out')},
            requirements=[ants_req.v('1.9')],
            mem_gb=16,
            wall_time=30)

        apply_trans_v = pipeline.add(
            'ApplyTransform_V_Atlas',
            ants.resampling.ApplyTransforms(
                interpolation='Linear',
                input_image_type=3,
                invert_transform_flags=[True, True, False]),
            inputs={
                'input_image': ('mni_template_vein_atlas', nifti_gz_format),
                'reference_image': ('qsm', nifti_gz_format),
                'transforms': (merge_trans, 'out')},
            requirements=[ants_req.v('1.9')],
            mem_gb=16,
            wall_time=30)

        # Run CV code
        pipeline.add(
            'cv_image',
            interface=CompositeVeinImage(),
            inputs={
                'mask': ('brain_mask', nifti_format),
                'qsm': ('qsm', nifti_format),
                'swi': ('swi', nifti_format),
                'q_prior': (apply_trans_q, 'output_image'),
                's_prior': (apply_trans_s, 'output_image'),
                'a_prior': (apply_trans_a, 'output_image'),
                'vein_atlas': (apply_trans_v, 'output_image')},
            outputs={
                'composite_vein_image': ('out_file', nifti_format)},
            requirements=[matlab_req.v('R2015a')],
            wall_time=300, mem_gb=24)

        return pipeline

    def shmrf_pipeline(self, **name_maps):

        pipeline = self.new_pipeline(
            name='shmrf_pipeline',
            name_maps=name_maps,
            desc="Compute Vein Mask using ShMRF",
            citations=[fsl_cite, matlab_cite])

        # Run ShMRF code
        pipeline.add(
            'shmrf',
            ShMRF(),
            inputs={
                'in_file': ('composite_vein_image', nifti_format),
                'mask': ('brain_mask', nifti_format)},
            outputs={
                'vein_mask': ('out_file', nifti_format)},
            requirements=[matlab_req.v('R2015a')],
            wall_time=30, mem_gb=16)

        return pipeline

    def cet_T2s(self, **options):

        pipeline = self.new_pipeline(
            name='CET_T2s',
            desc=("Construct cerebellum mask using SUIT template"),
            default_options={
                'SUIT_mask': self._lookup_template_mask_path('SUIT')},
            citations=[fsl_cite],
            options=options)

        # Initially use MNI space to warp SUIT mask into T2s space
        merge_trans = pipeline.add(
            'merge_transforms',
            Merge(3),
            inputs={
                'in3': (self._lookup_nl_tfm_inv_name('SUIT'), nifti_gz_format),
                'in2': (self._lookup_l_tfm_to_name('SUIT'), nifti_gz_format),
                'in1': ('T2s_to_T1_mat', text_matrix_format)})

        apply_trans = pipeline.add(
            'ApplyTransform',
            ants.resampling.ApplyTransforms(
                interpolation='NearestNeighbor',
                input_image_type=3,
                invert_transform_flags=[True, True, False],
                input_image=pipeline.option('SUIT_mask')),
            inputs={
                'transforms': (merge_trans, 'out'),
                'reference_image': ('betted_T2s', nifti_gz_format)},
            outputs={
                'cetted_T2s_mask': ('output_image', nifti_gz_format)},
            requirements=[ants_req.v('1.9')], mem_gb=16,
            wall_time=120)

        # Combine masks
        maths1 = pipeline.add(
            'combine_masks',
            fsl.utils.ImageMaths(
                suffix='_optiBET_masks',
                op_string='-mas',
                output_type='NIFTI_GZ'),
            inputs={
                'in_file': ('betted_T2s_mask', nifti_gz_format),
                'in_file2': (apply_trans, 'output_image')},
            requirements=[fsl_req.v('5.0.8')], mem_gb=16,
            wall_time=5)

        # Mask out t2s image
        pipeline.add(
            'mask_t2s',
            fsl.utils.ImageMaths(
                suffix='_optiBET_cerebellum',
                op_string='-mas',
                output_type='NIFTI_GZ'),
            inputs={
                'in_file': ('betted_T2s', nifti_gz_format),
                'in_file2': (maths1, 'output_image')},
            outputs={
                'cetted_T2s': ('out_file', nifti_gz_format)},
            requirements=[fsl_req.v('5.0.8')], mem_gb=16,
            wall_time=5)

        pipeline.add(
            'mask_t2s_last_echo',
            fsl.utils.ImageMaths(
                suffix='_optiBET_cerebellum',
                op_string='-mas',
                output_type='NIFTI_GZ'),
            inputs={
                'in_file': ('betted_T2s_last_echo', nifti_gz_format),
                'in_file2': (maths1, 'output_image')},
            outputs={
                'cetted_T2s_last_echo': ('out_file', nifti_gz_format)},
            requirements=[fsl_req.v('5.0.8')], mem_gb=16,
            wall_time=5)

        return pipeline

    def bet_T2s(self, **options):

        pipeline = self.new_pipeline(
            name='BET_T2s',
            desc=("python implementation of BET"),
            default_options={},
            citations=[fsl_cite],
            options=options)

        bet = pipeline.add(
            'bet',
            fsl.BET(
                frac=0.1,
                mask=True,
                output_type='NIFTI_GZ'),
            inputs={
                'in_file': ('t2s', nifti_gz_format)},
            outputs={
                'betted_T2s': ('out_file', nifti_gz_format),
                'betted_T2s_mask': ('mask_file', nifti_gz_format)},
            requirements=[fsl_req.v('5.0.8')], mem_gb=8,
            wall_time=45)

        pipeline.add(
            'mask',
            fsl.utils.ImageMaths(
                suffix='_BET_brain',
                op_string='-mas',
                output_type='NIFTI_GZ'),
            inputs={
                'in_file': ('t2s_last_echo', nifti_gz_format),
                'in_file2': (bet, 'mask_file')},
            outputs={
                'betted_T2s_last_echo': ('out_file', nifti_gz_format)},
            requirements=[fsl_req.v('5.0.8')], mem_gb=16,
            wall_time=5)

        return pipeline
