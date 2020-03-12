import re
from logging import getLogger
from nipype.interfaces import fsl, ants
from nipype.interfaces.utility import Select, Function
from arcana.analysis import AnalysisMetaClass
from arcana.data import FilesetSpec, InputFilesetSpec
from arcana.utils import get_class_info
from arcana.utils.interfaces import ListDir, CopyToDir
from arcana.analysis import ParamSpec, SwitchSpec
from arcana.utils.interfaces import Merge
from banana.interfaces.vein_analysis import (
    CompositeVeinImage, ShMRF)
from banana.interfaces.sti import (
    UnwrapPhase, VSharp, QsmILSQR, QsmStar, BatchUnwrapPhase, BatchVSharp,
    BatchQsmILSQR)
from banana.interfaces.phase import HipCombineChannels, Swi
from banana.interfaces.mask import (
    DialateMask, MaskCoils, MedianInMasks)
from banana.requirement import (fsl_req, matlab_req, ants_req, sti_req)
from banana.citation import (
    fsl_cite, matlab_cite, sti_cites)
from banana.file_format import (
    nifti_gz_format, nifti_format, text_matrix_format,
    multi_nifti_gz_format, STD_IMAGE_FORMATS)
from banana.reference import LocalReferenceData
from .base import MriAnalysis

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


class T2starAnalysis(MriAnalysis, metaclass=AnalysisMetaClass):

    desc = "T2*-weighted MRI contrast"

    add_data_specs = [
        # Set the magnitude to be generated from the preprocess_channels
        # pipeline
        FilesetSpec('magnitude', nifti_gz_format,
                    'kspace_recon_pipeline',
                    desc=("Generated from separate channel signals, "
                          "provided to 'channels'.")),
        # QSM and phase processing
        FilesetSpec('phase_mask', nifti_gz_format, 'phase_preproc_pipeline',
                    desc=("An agressive brain mask used to remove boundary "
                          "effect artefacts from QSM images")),
        FilesetSpec('tissue_phase', nifti_gz_format, 'phase_preproc_pipeline',
                    desc=("Phase image of brain tissue masked by "
                          "'phase_mask'")),
        FilesetSpec('q', nifti_gz_format, 'phase_preproc_pipeline',
                    desc=("Quality check on coil combination")),
        FilesetSpec('r2star', nifti_gz_format, 'phase_preproc_pipeline',
                    desc=("R2* contrast image")),
        FilesetSpec('swi', nifti_gz_format, 'swi_pipeline'),
        FilesetSpec('qsm', nifti_gz_format, 'qsm_pipeline',
                    desc=("Quantitative susceptibility image resolved "
                          "from T2* coil images")),
        # Vein analysis
        FilesetSpec('composite_vein_image', nifti_gz_format, 'cv_pipeline'),
        FilesetSpec('vein_mask', nifti_gz_format, 'shmrf_pipeline'),
        # Templates
        InputFilesetSpec('mni_template_qsm_prior', STD_IMAGE_FORMATS,
                         frequency='per_dataset',
                         default=LocalReferenceData('QSMPrior',
                                                    nifti_gz_format)),
        InputFilesetSpec('mni_template_swi_prior', STD_IMAGE_FORMATS,
                         frequency='per_dataset',
                         default=LocalReferenceData('SWIPrior',
                                                    nifti_gz_format)),
        InputFilesetSpec('mni_template_atlas_prior', STD_IMAGE_FORMATS,
                         frequency='per_dataset',
                         default=LocalReferenceData('VeinFrequencyPrior',
                                                    nifti_gz_format)),
        InputFilesetSpec('mni_template_vein_atlas', STD_IMAGE_FORMATS,
                         frequency='per_dataset',
                         default=LocalReferenceData('VeinFrequencyMap',
                                                    nifti_gz_format))]

    add_param_specs = [
        MriAnalysis.param_spec('reorient_to_std').with_new_default(False),
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
        ParamSpec('bet_g_threshold', 0.0),
        ParamSpec('swi_power', 4, desc=(
            "The power which the masked phase image is raised to in the SWI "
            "calculation"))]

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

    def phase_preproc_pipeline(self, **name_maps):

        pipeline = self.new_pipeline(
            name='phase_preproc_pipeline',
            name_maps=name_maps,
            desc="Combines coil_channels, unwraps phase and masks phase",
            citations=[sti_cites, fsl_cite, matlab_cite])

        # Combine channels to produce phase and magnitude images
        channel_combine = pipeline.add(
            'channel_combine',
            HipCombineChannels(),
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

        # Remove background noise and tidy up phase mask
        pipeline.add(
            "vsharp",
            VSharp(
                mask_manip="imerode({{}}>0, ball({}))".format(
                    self.parameter('qsm_me_erosion_size')),
                single_comp_thread=False),
            inputs={
                'voxelsize': ('voxel_sizes', float),
                'in_file': (unwrap, 'out_file'),
                'mask': ('brain_mask', nifti_gz_format)},
            outputs={
                'phase_mask': ('new_mask', nifti_gz_format),
                'tissue_phase': ('out_file', nifti_format)},
            requirements=[matlab_req.v('r2018a'), sti_req.v(3.0)])

        return pipeline

    def qsm_pipeline(self, **name_maps):

        pipeline = self.new_pipeline(
            name='qsm_pipeline',
            name_maps=name_maps,
            desc=("Calculates Quantitative Susceptibility Mappings (QSM) "
                  "from tissue phase"),
            citations=[sti_cites, fsl_cite, matlab_cite])

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
            QsmStar(
                padsize=self.parameter('qsm_padding'),
                mask_manip="{}>0",
                single_comp_thread=False),
            inputs={
                'in_file': ('tissue_phase', nifti_format),
                'mask': ('phase_mask', nifti_format),
                'voxelsize': ('voxel_sizes', float),
                'TE': (delta_te, 'delta_te'),
                'B0': ('main_field_strength', float),
                'H': ('main_field_orient', float)},
            outputs={
                'qsm': ('out_file', nifti_format)},
            requirements=[matlab_req.v('r2018a'), sti_req.v(3.0)])

        return pipeline

    def swi_pipeline(self, **name_maps):

        pipeline = self.new_pipeline(
            name='swi_pipeline',
            name_maps=name_maps,
            desc=("Calculates Susceptibility-weighted images (SWI) from tissue"
                  " phase and magnitude images"),
            citations=[])
        # https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.20198

        # Run SWI
        pipeline.add(
            'swi',
            Swi(
                alpha=self.parameter('swi_power')),
            inputs={
                'tissue_phase': ('tissue_phase', nifti_gz_format),
                'magnitude': ('magnitude', nifti_gz_format),
                'mask': ('phase_mask', nifti_gz_format)},
            outputs={
                'swi': ('out_file', nifti_gz_format)})

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

