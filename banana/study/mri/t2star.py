import re
from logging import getLogger
from nipype.interfaces import fsl, ants
from nipype.interfaces.utility import Select, Function
from arcana.study import StudyMetaClass
from arcana.data import FilesetSpec, InputFilesetSpec
from arcana.utils import get_class_info
from arcana.utils.interfaces import ListDir, CopyToDir
from arcana.study import ParamSpec, SwitchSpec
from arcana.utils.interfaces import Merge
from banana.interfaces.vein_analysis import (
    CompositeVeinImage, ShMRF)
from banana.interfaces.sti import (
    UnwrapPhase, VSharp, QSMiLSQR, QSMStar, BatchUnwrapPhase, BatchVSharp,
    BatchQSMiLSQR)
from banana.interfaces.coils import HIPCombineChannels, ToPolarCoords
from banana.interfaces.mask import (
    DialateMask, MaskCoils, MedianInMasks)
from banana.requirement import (fsl_req, matlab_req, ants_req, sti_req)
from banana.citation import (
    fsl_cite, matlab_cite, sti_cites)
from banana.file_format import (
    nifti_gz_format, nifti_format, text_matrix_format,
    multi_nifti_gz_format, STD_IMAGE_FORMATS)
from banana.reference import LocalReferenceData
from .base import MriStudy

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


def calculate_delta_te(echo_times):
    "Get the time difference between echos in miliseconds"
    return (echo_times[1] - echo_times[0]) * 1000


class T2starStudy(MriStudy, metaclass=StudyMetaClass):

    desc = "T2*-weighted MRI contrast"

    add_data_specs = [
        # Set the magnitude to be generated from the preprocess_channels
        # pipeline
        FilesetSpec('magnitude', nifti_gz_format,
                    'kspace_recon_pipeline',
                    desc=("Generated from separate channel signals, "
                          "provided to 'channels'.")),
        # QSM and phase processing
        FilesetSpec('swi', nifti_gz_format, 'swi_pipeline'),
        FilesetSpec('qsm', nifti_gz_format, 'qsm_pipeline',
                    desc=("Quantitative susceptibility image resolved "
                          "from T2* coil images")),
        FilesetSpec('q', nifti_gz_format, 'qsm_pipeline',
                    desc=("Quality check on coil combination")),
        FilesetSpec('r2star', nifti_gz_format, 'qsm_pipeline',
                    desc=("R2* contrast image")),
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
        ParamSpec('qsm_num_echos', 1),
        MriStudy.param_spec('reorient_to_std').with_new_default(False),
        ParamSpec('qsm_echo', 1,
                  desc=("Which echo (by index starting at 1) to use when "
                        "using single echo")),
        ParamSpec('qsm_padding', [12, 12, 12]),
        ParamSpec('qsm_mask_dialation', [11, 11, 11]),
        ParamSpec('qsm_me_erosion_size', 4),
        ParamSpec('qsm_se_erosion_size', 10),
        SwitchSpec('bet_robust', False),
        SwitchSpec('bet_robust', False),
        ParamSpec('bet_f_threshold', 0.1),
        ParamSpec('bet_g_threshold', 0.0)]

    def kspace_recon_pipeline(self, **name_maps):
        pipeline = super().kspace_recon_pipeline(**name_maps)

        # Bias correct the output magnitude image
        pipeline.add(
            'bias_correct',
            ants.N4BiasFieldCorrection(
                dimension=3),
            inputs={
                'input_image': (pipeline.node('grappa'), 'out_file')},
            outputs={
                'magnitude': ('output_image', nifti_gz_format)},
            requirements=[ants_req.v('2.2.0')])

        return pipeline

    def qsm_pipeline(self, **name_maps):

        pipeline = self.new_pipeline(
            name='qsm_pipeline',
            name_maps=name_maps,
            desc="Resolve QSM from t2star coils",
            citations=[sti_cites, fsl_cite, matlab_cite])

        # Combine channels to produce phase and magnitude images
        channel_combine = pipeline.add(
            'channel_combine',
            HIPCombineChannels(),
            inputs={
                'channels_dir': ('channels', multi_nifti_gz_format),
                'echo_times': ('echo_times', float)},
            outputs={
                'r2star': ('r2star', nifti_gz_format),
                'q': ('q', nifti_gz_format)})

        # Unwrap phase using Laplacian unwrapping
        unwrap = pipeline.add(
            'unwrap',
            UnwrapPhase(
                padsize=self.parameter('qsm_padding'),
                single_comp_thread=False),
            inputs={
                'voxelsize': ('voxel_sizes', float),
                'in_file': (channel_combine, 'phase')},
            requirements=[matlab_req.v('r2018a'), sti_req.v(3.0)])

        # Remove background noise
        vsharp = pipeline.add(
            "vsharp",
            VSharp(
                mask_manip="imerode({{}}>0, ball({}))".format(
                    self.parameter('qsm_me_erosion_size')),
                single_comp_thread=False),
            inputs={
                'voxelsize': ('voxel_sizes', float),
                'in_file': (unwrap, 'out_file'),
                'mask': ('brain_mask', nifti_gz_format)},
            requirements=[matlab_req.v('r2018a'), sti_req.v(3.0)])

        delta_te = pipeline.add(
            "delta_te",
            Function(
                input_names=['echo_times'],
                output_names=['delta_te'],
                function=calculate_delta_te),
            inputs={
                'echo_times': ('echo_times', float)})

        # Run QSM star
        pipeline.add(
            'qsmrecon',
            QSMStar(
                padsize=self.parameter('qsm_padding'),
                mask_manip="{}>0",
                single_comp_thread=False),
            inputs={
                'voxelsize': ('voxel_sizes', float),
                'mask': (vsharp, 'new_mask'),
                'TE': (delta_te, 'delta_te'),
                'B0': ('main_field_strength', float),
                'H': ('main_field_orient', float),
                'in_file': (vsharp, 'out_file')},
            outputs={
                'qsm': ('out_file', nifti_gz_format)},
            requirements=[matlab_req.v('r2018a'), sti_req.v(3.0)])

        return pipeline

    def _construct_single_echo_qsm_pipeline(self, pipeline):

        erosion = pipeline.add(
            'mask_erosion',
            fsl.ErodeImage(
                kernel_shape='sphere',
                kernel_size=self.parameter('qsm_se_erosion_size'),
                output_type='NIFTI'),
            inputs={
                'in_file': ('brain_mask', nifti_gz_format)},
            requirements=[fsl_req.v('5.0.8')],
            wall_time=15, mem_gb=12)

        # Dialate eroded mask
        dialate = pipeline.add(
            'dialate',
            DialateMask(
                dialation=self.parameter('qsm_mask_dialation')),
            inputs={
                'in_file': (erosion, 'out_file')},
            requirements=[matlab_req.v('r2017a')])

        to_polar = pipeline.add(
            'to_polar',
            ToPolarCoords(
                in_fname_re=self.parameter('channel_fname_regex'),
                real_label=self.parameter('channel_real_label'),
                imaginary_label=self.parameter('channel_imag_label')),
            inputs={
                'in_dir': ('channels', multi_nifti_gz_format)})

        # List files for the phases of separate channel
        list_phases = pipeline.add(
            'list_phases',
            ListDir(
                sort_key=coil_sort_key,
                filter=CoilEchoFilter(self.parameter('qsm_echo'))),
            inputs={
                'directory': (to_polar, 'phases_dir')})

        # List files for the phases of separate channel
        list_mags = pipeline.add(
            'list_mags',
            ListDir(
                sort_key=coil_sort_key,
                filter=CoilEchoFilter(self.parameter('qsm_echo'))),
            inputs={
                'directory': (to_polar, 'mag_channels')})

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
            requirements=[matlab_req.v('r2017a'), sti_req.v(3.0)])

        # Background phase removal
        vsharp = pipeline.add(
            "vsharp",
            BatchVSharp(
                mask_manip='{}>0'),
            inputs={
                'voxelsize': ('voxel_sizes', float),
                'mask': (mask_coils, 'out_files'),
                'in_file': (unwrap, 'out_file')},
            requirements=[matlab_req.v('r2017a'), sti_req.v(3.0)])

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
            requirements=[matlab_req.v('r2017a'), sti_req.v(3.0)],
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

    def swi_pipeline(self, **name_maps):

        raise NotImplementedError

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
