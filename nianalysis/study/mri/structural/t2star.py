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

    def qsm_pipeline(self, **nmaps):
        """
        Process dual echo data for QSM (TE=[7.38, 22.14])

        NB: Default values come from the STI-Suite
        """
        pipeline = self.pipeline(
            name='qsm_pipeline',
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
            citations=[sti_cites, fsl_cite, matlab_cite],
            **nmaps)

        erosion = pipeline.create_node(interface=fsl.ErodeImage(),
                                       name='mask_erosion',
                                       requirements=[fsl5_req],
                                       wall_time=15, memory=12000)
        erosion.inputs.kernel_shape = 'sphere'
        erosion.inputs.kernel_size = 2
        pipeline.connect_input('brain_mask', erosion, 'in_file')

        # Copy geometry from scanner image to QSM
        qsm_geom = pipeline.create_node(
            fsl.CopyGeom(),
            name='qsm_copy_geometry',
            requirements=[fsl5_req],
            memory=4000,
            wall_time=5)
        pipeline.connect_input('header_image', qsm_geom, 'in_file')

        # If we have multiple echoes we can combine the phase images from
        # each channel into a single image. Otherwise for single echo sequences
        # we need to perform QSM on each coil separately and then combine
        # afterwards.
        if self.branch('qsm_dual_echo'):
            # Combine channels to produce phase and magnitude images
            channel_combine = pipeline.create_node(
                interface=HIPCombineChannels(), name='channel_combine')
            pipeline.connect_input('channel_mags', channel_combine,
                                   'magnitudes_dir')
            pipeline.connect_input('channel_phases', channel_combine,
                                   'phases_dir')
            # Unwrap phase using Laplacian unwrapping
            unwrap = pipeline.create_node(UnwrapPhase(), name='unwrap')
            pipeline.connect(channel_combine, 'phase', unwrap, 'in_file')
            # Remove background noise
            vsharp = pipeline.add("vsharp", VSharp())
            pipeline.connect(erosion, 'out_file', vsharp, 'mask')
            vsharp.inputs.mask_manip = "imerode({}>0, ball(5))"
            # Run QSM iLSQR
            qsmrecon = pipeline.create_node(QSMiLSQR(), name='qsmrecon')
            pipeline.connect(qsmrecon, 'out_file', qsm_geom, 'dest_file')
        else:
            def coil_echo_filter(fname):
                return re.match(r'coil_\d+_(\d+)',
                                fname) == self.parameter('qsm_echo')
            # Dialate eroded mask
            dialate = pipeline.create_node(DialateMask(), name='dialate')
            pipeline.connect(erosion, 'out_file', dialate, 'in_file')
            dialate.inputs.dialation = self.parameter('qsm_mask_dialation')
            # List files for the phases of separate channel
            list_phases = pipeline.create_node(ListDir(), name='list_phases')
            pipeline.connect_input('channel_phases', list_phases, 'directory')
            list_phases.inputs.sort_key = coil_sort_key
            list_phases.inputs.filter = coil_echo_filter
            # List magnitude in channel magnitudes
            list_mags = pipeline.create_node(ListDir(), name='list_mags')
            pipeline.connect_input('channel_mags', list_mags, 'directory')
            list_phases.inputs.sort_key = coil_sort_key
            list_phases.inputs.filter = coil_echo_filter
            # Generate coil specific masks
            coil_masks = pipeline.create_map_node(
                CoilMask(), name='coil_masks', iterfield=['in_file'])
            pipeline.connect(dialate, 'out_file', coil_masks,
                             'whole_brain_mask')
            # Unwrap phase
            unwrap = pipeline.create_map_node(UnwrapPhase(), name='unwrap',
                                              iterfield=['in_file'])
            pipeline.connect(list_phases, 'files', unwrap, 'in_file')
            # Background phase removal
            vsharp = pipeline.create_map_node(VSharp(), name="vsharp",
                                              iterfield=['in_file', 'mask'])
            pipeline.connect(coil_masks, 'out_file', vsharp, 'mask')
            vsharp.inputs.mask_manip = '{}>0'
            # Perform channel-wise QSM
            qsmrecon = pipeline.create_map_node(QSMiLSQR(), name='qsm',
                                                iterfield=['in_file', 'mask'])
            combine_qsm = pipeline.create_node(MedianInMasks(),
                                               name='combine_qsm')
            pipeline.connect(qsmrecon, 'out_file', combine_qsm, 'channels')
            pipeline.connect(vsharp, 'new_mask', combine_qsm, 'channel_masks')
            pipeline.connect(dialate, 'out_file', combine_qsm,
                             'whole_brain_mask')
            pipeline.connect(combine_qsm, 'out_file', qsm_geom, 'dest_file')
        # Set common parameters and connections for QSM pipeline
        pipeline.connect_input('voxel_sizes', unwrap, 'voxelsize')
        unwrap.inputs.padsize = self.parameter('qsm_padding')
        pipeline.connect(unwrap, 'out_file', vsharp, 'in_file')
        pipeline.connect_input('voxel_sizes', vsharp, 'voxelsize')
        pipeline.connect(vsharp, 'out_file', qsmrecon, 'in_file')
        pipeline.connect(vsharp, 'new_mask', qsmrecon, 'mask')
        qsmrecon.inputs.mask_manip = "{}>0"
        pipeline.connect_input('voxel_sizes', qsmrecon, 'voxelsize')
        pipeline.connect_input('echo_times', qsmrecon, 'te')
        pipeline.connect_input('main_field_strength', qsmrecon, 'B0')
        pipeline.connect_input('main_field_orient', qsmrecon, 'H')
        qsmrecon.inputs.padsize = self.parameter('qsm_padding')
        return pipeline

    def swi_pipeline(self, **kwargs):
        pipeline = self.pipeline(
            name='swi',
            inputs=[FilesetSpec('magnitude', nifti_gz_format),
                    FilesetSpec('channel_phases', multi_nifti_gz_format)],
            outputs=[FilesetSpec('swi', nifti_gz_format)],
            desc=("Calculate susceptibility-weighted image from magnitude and "
                  "phase"),
            references=[],
            **kwargs)
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

    def composite_vein_pipeline(self, **kwargs):

        pipeline = self.pipeline(
            name='comp_vein_image_pipeline',
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
            citations=[fsl_cite, matlab_cite],
            **kwargs)

        # Prepare SWI flip(flip(swi,1),2)
        flip = pipeline.create_node(interface=qsm.FlipSWI(), name='flip_swi',
                                    requirements=[matlab2015_req],
                                    wall_time=10, memory=16000)
        pipeline.connect_input('t2star_swi', flip, 'in_file')
        pipeline.connect_input('t2star_qsm', flip, 'hdr_file')

        # Interpolate priors and atlas
        merge_trans = pipeline.create_node(
            utils.Merge(3), name='merge_transforms')
        pipeline.connect_input('t2star_coreg_matrix', merge_trans, 'in1')
        pipeline.connect_input('t1_coreg_to_atlas_mat', merge_trans, 'in2')
        pipeline.connect_input('t1_coreg_to_atlas_warp', merge_trans, 'in3')

        apply_trans_q = pipeline.create_node(
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

        apply_trans_s = pipeline.create_node(
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

        apply_trans_a = pipeline.create_node(
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

        apply_trans_v = pipeline.create_node(
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
        cv_image = pipeline.create_node(
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

    def shmrf_pipeline(self, **kwargs):

        pipeline = self.pipeline(
            name='shmrf_pipeline',
            inputs=[FilesetSpec('composite_vein_image', nifti_gz_format),
                    FilesetSpec('t2star_brain_mask', nifti_gz_format)],
            outputs=[FilesetSpec('vein_mask', nifti_gz_format)],
            desc="Compute Vein Mask using ShMRF",
            citations=[fsl_cite, matlab_cite],
            **kwargs)

        # Run ShMRF code
        shmrf = pipeline.create_node(interface=qsm.ShMRF(), name='shmrf',
                                     requirements=[matlab2015_req],
                                     wall_time=30, memory=16000)
        pipeline.connect_input('composite_vein_image', shmrf, 'in_file')
        pipeline.connect_input('t2star_brain_mask', shmrf, 'mask_file')

        # Output vein map
        pipeline.connect_output('vein_mask', shmrf, 'out_file')

        return pipeline
