import os.path as op
import re
from arcana.study import (
    StudyMetaClass, MultiStudy, MultiStudyMetaClass, SubStudySpec)
from arcana.data import FilesetSpec, FilesetCollection, Fileset, FieldSpec
from nianalysis.requirement import (fsl5_req, matlab2015_req,
                                    ants19_req)
from nianalysis.citation import (
    fsl_cite, matlab_cite, sti_cites)
from nianalysis.file_format import (
    nifti_gz_format, text_matrix_format, dicom_format, multi_nifti_gz_format)
from nianalysis.interfaces import qsm
from arcana.interfaces import utils
from ..base import MRIStudy
from .t1 import T1Study
from nipype.interfaces import fsl, ants
from arcana.interfaces.utils import ListDir
from nianalysis.interfaces.sti import UnwrapPhase, VSharp, QSMiLSQR
from nianalysis.interfaces.custom.coils import HIPCombineChannels
from nianalysis.interfaces.custom.mask import (
    DialateMask, CoilMask, MedianInMasks)
import nianalysis
from arcana.parameter import ParameterSpec, SwitchSpec

atlas_path = op.abspath(op.join(op.dirname(nianalysis.__file__), 'atlases'))


def coil_sort_key(fname):
    return re.match(r'coil_(\d+)_\d+\.nii\.gz', fname).group(1)


class QsmAtlas(FilesetCollection):

    def __init__(self, name):
        super().__init__(
            name,
            [Fileset.from_path(op.join(atlas_path, '{}.nii.gz'.format(name)),
                               frequency='per_study')])


class T2StarStudy(MRIStudy, metaclass=StudyMetaClass):

    add_data_specs = [
        # Set the magnitude to be generated from the prepare_channels
        # pipeline
        FilesetSpec('magnitude', nifti_gz_format, 'prepare_channels',
                    desc=("Generated from separate channel signals, "
                          "provided to 'coil_channels'.")),
        # QSM and phase processing
        FilesetSpec('swi', nifti_gz_format, 'swi_pipeline'),
        FilesetSpec('qsm', nifti_gz_format, 'qsm_pipeline',
                    desc=("Quantitative susceptibility image resolved "
                                 "from T2* coil images")),
        FilesetSpec('header_image', dicom_format, desc=(
            "The image that contains the header information required to "
            "perform the analysis (e.g. TE, B0, H). Alternatively, values "
            "for extracted fields can be explicitly passed as inputs to the "
            "Study"))]

    add_parameter_specs = [
        SwitchSpec('qsm_dual_echo', False),
        ParameterSpec(
            'qsm_echo', 0,
            desc="Which echo (by index) to use when using single echo"),
        ParameterSpec('qsm_padding', [12, 12, 12]),
        ParameterSpec('qsm_mask_dialation', [11, 11, 11])]

    def header_extraction_pipeline(self, **kwargs):
        return self.header_extraction_pipeline_factory(
            'header_info_extraction', 'header_image', **kwargs)

    def qsm_pipeline(self, **mods):
        """
        Process dual echo data for QSM (TE=[7.38, 22.14])

        NB: Default values come from the STI-Suite
        """
        pipeline = self.pipeline(
            name='qsm_pipeline',
            modifications=mods,
            inputs=[FilesetSpec('channel_phases', multi_nifti_gz_format),
                    FilesetSpec('channel_mags', multi_nifti_gz_format),
                    FilesetSpec('magnitude', nifti_gz_format),
                    FilesetSpec('brain_mask', nifti_gz_format),
                    FilesetSpec('header_image', dicom_format),
                    FieldSpec('voxel_sizes', float),
                    FieldSpec('echo_times', float),
                    FieldSpec('main_field_strength', float),
                    FieldSpec('main_field_orient', float)],
            outputs=[FilesetSpec('qsm', nifti_gz_format)],
            desc="Resolve QSM from t2star coils",
            references=[sti_cites, fsl_cite, matlab_cite])

        erosion = pipeline.add(
            'mask_erosion',
            fsl.ErodeImage(
                kernel_shape='sphere',
                kernel_size=2),
            inputs={
                'in_file': ('brain_mask', nifti_gz_format)},
            requirements=[fsl5_req],
            wall_time=15, memory=12000)

        # Copy geometry from scanner image to QSM
        qsm_geom = pipeline.add(
            'qsm_copy_geometry',
            fsl.CopyGeom(),
            inputs={
                'in_file': ('header_image', dicom_format)},
            outputs={
                'out_file': ('qsm', nifti_gz_format)},
            requirements=[fsl5_req],
            memory=4000,
            wall_time=5)

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
                    'voxelsize': 'voxel_sizes'},
                connections={
                    'in_file': (channel_combine, 'phase')})

            # Remove background noise
            vsharp = pipeline.add(
                "vsharp",
                VSharp(
                    mask_manip="imerode({}>0, ball(5))"),
                inputs={
                    'voxelsize': 'voxel_sizes'},
                connections={
                    'in_file': (unwrap, 'out_file'),
                    'mask': (erosion, 'out_file')})

            # Run QSM iLSQR
            qsmrecon = pipeline.add(
                'qsmrecon',
                QSMiLSQR(
                    mask_manip="{}>0",
                    padsize=self.parameter('qsm_padding')),
                inputs={
                    'voxelsize': 'voxel_sizes',
                    'te': 'echo_times',
                    'B0': 'main_field_strength',
                    'H': 'main_field_orient'},
                connections={
                    'in_file': (vsharp, 'out_file'),
                    'mask': (vsharp, 'new_mask')})

            # Connect to final node, which adds geometry from header
            self.connect(qsmrecon, 'out_file', qsm_geom, 'dest_file')
        else:
            def coil_echo_filter(fname):
                return re.match(r'coil_\d+_(\d+)',
                                fname) == self.parameter('qsm_echo')
            # Dialate eroded mask
            dialate = pipeline.add(
                'dialate',
                DialateMask(
                    dialation=self.parameter('qsm_mask_dialation')),
                connections={
                    'in_file': (erosion, 'out_file')})

            # List files for the phases of separate channel
            list_phases = pipeline.add(
                'list_phases',
                ListDir(
                    sort_key=coil_sort_key,
                    filter=coil_echo_filter),
                inputs={
                    'directory': ('channel_phases', multi_nifti_gz_format)})

            # List magnitude in channel magnitudes
            list_mags = pipeline.add(
                'list_mags',
                ListDir(
                    sort_key=coil_sort_key,
                    filter=coil_echo_filter),
                inputs={
                    'directory': ('channel_mags', multi_nifti_gz_format)})

            # Generate coil specific masks
            coil_masks = pipeline.add(
                'coil_masks',
                CoilMask(),
                connections={
                    'whole_brain_mask': (dialate, 'out_file')},
                iterfield=['in_file'])

            # Unwrap phase
            unwrap = pipeline.add(
                'unwrap',
                UnwrapPhase(
                    padsize=self.parameter('qsm_padding')),
                inputs={
                    'voxelsize': 'voxel_sizes'},
                connections={
                    'in_file': (list_phases, 'files')},
                iterfield=['in_file'])

            # Background phase removal
            vsharp = pipeline.add(
                "vsharp",
                VSharp(
                    mask_manip='{}>0'),
                inputs={
                    'voxelsize': 'voxel_sizes'},
                connections={
                    'mask': (coil_masks, 'out_file'),
                    'in_file': (unwrap, 'out_file')},
                iterfield=['in_file', 'mask'])

            # Perform channel-wise QSM
            qsmrecon = pipeline.add(
                'qsm',
                QSMiLSQR(
                    mask_manip="{}>0",
                    padsize=self.parameter('qsm_padding')),
                inputs={
                    'voxelsize': 'voxel_sizes',
                    'te': 'echo_times',
                    'B0': 'main_field_strength',
                    'H': 'main_field_orient'},
                connections={
                    'in_file': (vsharp, 'out_file'),
                    'mask': (vsharp, 'new_mask')},
                iterfield=['in_file', 'mask'])

            # Combine channel QSM by taking the median
            combine_qsm = pipeline.add(
                'combine_qsm',
                MedianInMasks(),
                connections={
                    'channels': (qsmrecon, 'out_file'),
                    'channel_masks': (vsharp, 'new_mask'),
                    'whole_brain_mask': (dialate, 'out_file')})

            # Connect to final node, which adds geometry from header
            self.connect(combine_qsm, 'out_file', qsm_geom, 'dest_file')

        return pipeline

    def swi_pipeline(self, **mods):
        pipeline = self.pipeline(
            name='swi',
            modifications=mods,
            inputs=[FilesetSpec('magnitude', nifti_gz_format),
                    FilesetSpec('channel_phases', multi_nifti_gz_format)],
            outputs=[FilesetSpec('swi', nifti_gz_format)],
            desc=("Calculate susceptibility-weighted image from magnitude and "
                  "phase"))
        # Not implemented yet.
        return pipeline


class T2StarT1Study(MultiStudy, metaclass=MultiStudyMetaClass):

    add_sub_study_specs = [
        SubStudySpec('t1', T1Study),
        SubStudySpec('t2star', T2StarStudy,
                     name_map={'t1_brain': 'coreg_ref_brain'})]

    add_data_specs = [
        # Vein analysis
        FilesetSpec('composite_vein_image', nifti_gz_format,
                    'composite_vein_pipeline'),
        FilesetSpec('vein_mask', nifti_gz_format, 'shmrf_pipeline'),
        # Templates
        FilesetSpec('mni_template_qsm_prior', nifti_gz_format,
                    default=QsmAtlas('QSMPrior')),
        FilesetSpec('mni_template_swi_prior', nifti_gz_format,
                    default=QsmAtlas('SWIPrior')),
        FilesetSpec('mni_template_atlas_prior', nifti_gz_format,
                    default=QsmAtlas('VeinFrequencyPrior')),
        FilesetSpec('mni_template_vein_atlas', nifti_gz_format,
                    default=QsmAtlas('VeinFrequencyMap'))]

#     add_parameter_specs = [
#         # Change the default atlast coreg tool to FNIRT
#         SwitchSpec('t1_atlas_coreg_tool', 'fnirt', ('fnirt', 'ants'))]

    def composite_vein_pipeline(self, **mods):

        pipeline = self.pipeline(
            name='comp_vein_image_pipeline',
            modifications=mods,
            inputs=[FilesetSpec('t2star_qsm', nifti_gz_format),
                    FilesetSpec('t2star_swi', nifti_gz_format),
                    FilesetSpec('t2star_brain_mask', nifti_gz_format),
                    FilesetSpec('t2star_coreg_matrix', text_matrix_format),
                    FilesetSpec('t1_coreg_to_atlas_mat', text_matrix_format),
                    FilesetSpec('t1_coreg_to_atlas_warp', nifti_gz_format),
                    FilesetSpec('mni_template_qsm_prior', nifti_gz_format),
                    FilesetSpec('mni_template_swi_prior', nifti_gz_format),
                    FilesetSpec('mni_template_atlas_prior', nifti_gz_format),
                    FilesetSpec('mni_template_vein_atlas', nifti_gz_format)],
            outputs=[FilesetSpec('composite_vein_image', nifti_gz_format)],
            desc="Compute Composite Vein Image",
            references=[fsl_cite, matlab_cite])

        # Prepare SWI flip(flip(swi,1),2)
        flip = pipeline.add(interface=qsm.FlipSWI(), name='flip_swi',
                                    requirements=[matlab2015_req],
                                    wall_time=10, memory=16000)
        pipeline.connect_input('t2star_swi', flip, 'in_file')
        pipeline.connect_input('t2star_qsm', flip, 'hdr_file')

        # Interpolate priors and atlas
        merge_trans = pipeline.add(
            utils.Merge(3), name='merge_transforms')
        pipeline.connect_input('t2star_coreg_matrix', merge_trans, 'in1')
        pipeline.connect_input('t1_coreg_to_atlas_mat', merge_trans, 'in2')
        pipeline.connect_input('t1_coreg_to_atlas_warp', merge_trans, 'in3')

        apply_trans_q = pipeline.add(
            ants.resampling.ApplyTransforms(),
            name='ApplyTransform_Q_Prior', requirements=[ants19_req],
            memory=16000, wall_time=30)
        apply_trans_q.inputs.interpolation = 'Linear'
        apply_trans_q.inputs.input_image_type = 3
        apply_trans_q.inputs.invert_transform_flags = [True, True, False]
        pipeline.connect_input(
            'mni_template_qsm_prior',
            apply_trans_q,
            'input_image')
        pipeline.connect(merge_trans, 'out', apply_trans_q, 'transforms')
        pipeline.connect_input('t2star_qsm', apply_trans_q, 'reference_image')

        apply_trans_s = pipeline.add(
            ants.resampling.ApplyTransforms(),
            name='ApplyTransform_S_Prior',
            requirements=[ants19_req], memory=16000, wall_time=30)
        apply_trans_s.inputs.interpolation = 'Linear'
        apply_trans_s.inputs.input_image_type = 3
        apply_trans_s.inputs.invert_transform_flags = [True, True, False]

        pipeline.connect_input(
            'mni_template_swi_prior',
            apply_trans_s,
            'input_image')
        pipeline.connect(merge_trans, 'out', apply_trans_s, 'transforms')
        pipeline.connect_input('t2star_qsm', apply_trans_s, 'reference_image')

        apply_trans_a = pipeline.add(
            ants.resampling.ApplyTransforms(),
            name='ApplyTransform_A_Prior', requirements=[ants19_req],
            memory=16000, wall_time=30)
        apply_trans_a.inputs.interpolation = 'Linear'
        apply_trans_a.inputs.input_image_type = 3
        apply_trans_a.inputs.invert_transform_flags = [True, True, False]

        pipeline.connect_input(
            'mni_template_atlas_prior',
            apply_trans_a,
            'input_image')
        pipeline.connect(merge_trans, 'out', apply_trans_a, 'transforms')
        pipeline.connect_input('t2star_qsm', apply_trans_a, 'reference_image')

        apply_trans_v = pipeline.add(
            ants.resampling.ApplyTransforms(),
            name='ApplyTransform_V_Atlas', requirements=[ants19_req],
            memory=16000, wall_time=30)
        apply_trans_v.inputs.interpolation = 'Linear'
        apply_trans_v.inputs.input_image_type = 3
        apply_trans_v.inputs.invert_transform_flags = [True, True, False]
        pipeline.connect_input(
            'mni_template_vein_atlas',
            apply_trans_v,
            'input_image')
        pipeline.connect(merge_trans, 'out', apply_trans_v, 'transforms')
        pipeline.connect_input('t2star_qsm', apply_trans_v, 'reference_image')

        # Run CV code
        cv_image = pipeline.add(
            interface=qsm.CVImage(),
            name='composite_vein_image',
            requirements=[matlab2015_req],
            wall_time=300, memory=24000)
        pipeline.connect_input('t2star_qsm', cv_image, 'qsm')
        # pipeline.connect_input('swi', composite_vein_image, 'swi')
        pipeline.connect(flip, 'out_file', cv_image, 'swi')
        pipeline.connect_input('t2star_brain_mask', cv_image, 'mask')
        pipeline.connect(apply_trans_q, 'output_image', cv_image, 'q_prior')
        pipeline.connect(apply_trans_s, 'output_image', cv_image, 's_prior')
        pipeline.connect(apply_trans_a, 'output_image', cv_image, 'a_prior')
        pipeline.connect(apply_trans_v, 'output_image', cv_image, 'vein_atlas')

        # Output final [0-1] map
        pipeline.connect_output('composite_vein_image', cv_image, 'out_file')

        return pipeline

    def shmrf_pipeline(self, **mods):

        pipeline = self.pipeline(
            name='shmrf_pipeline',
            modifications=mods,
            inputs=[FilesetSpec('composite_vein_image', nifti_gz_format),
                    FilesetSpec('t2star_brain_mask', nifti_gz_format)],
            outputs=[FilesetSpec('vein_mask', nifti_gz_format)],
            desc="Compute Vein Mask using ShMRF",
            references=[fsl_cite, matlab_cite])

        # Run ShMRF code
        shmrf = pipeline.add(interface=qsm.ShMRF(), name='shmrf',
                                     requirements=[matlab2015_req],
                                     wall_time=30, memory=16000)
        pipeline.connect_input('composite_vein_image', shmrf, 'in_file')
        pipeline.connect_input('t2star_brain_mask', shmrf, 'mask_file')

        # Output vein map
        pipeline.connect_output('vein_mask', shmrf, 'out_file')

        return pipeline
