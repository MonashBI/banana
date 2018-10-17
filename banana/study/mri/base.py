from nipype.interfaces import fsl
from nipype.interfaces.spm.preprocess import Coregister
from banana.requirement import spm12_req
from banana.citation import spm_cite
from banana.file_format import (
    nifti_format, motion_mats_format, directory_format, nifti_gz_format,
    multi_nifti_gz_format, zip_format, STD_IMAGE_FORMATS)
from arcana.data import FilesetSpec, FieldSpec, AcquiredFilesetSpec
from arcana.study.base import Study, StudyMetaClass
from banana.citation import fsl_cite, bet_cite, bet2_cite
from banana.file_format import (
    dicom_format, text_format, gif_format)
from nipype.interfaces.utility import IdentityInterface
from banana.requirement import fsl5_req, mrtrix3_req, fsl509_req, ants2_req
from nipype.interfaces.fsl import (FLIRT, FNIRT, Reorient2Std)
from arcana.exception import ArcanaUsageError
from banana.interfaces.mrtrix.transform import MRResize
from banana.interfaces.custom.dicom import (DicomHeaderInfoExtraction)
from nipype.interfaces.utility import Split, Merge
from banana.interfaces.fsl import FSLSlices
from banana.file_format import text_matrix_format
import logging
from banana.interfaces.ants import AntsRegSyn
from banana.interfaces.custom.coils import ToPolarCoords
from arcana.interfaces.utils import ListDir, CopyToDir
from nipype.interfaces.ants.resampling import ApplyTransforms
from arcana.parameter import ParameterSpec, SwitchSpec
from banana.interfaces.custom.motion_correction import (
    MotionMatCalculation)
from banana.atlas import FslAtlas

logger = logging.getLogger('arcana')


class MriStudy(Study, metaclass=StudyMetaClass):

    add_data_specs = [
        AcquiredFilesetSpec('magnitude', STD_IMAGE_FORMATS,
                            desc=("Typically the primary scan acquired from "
                                  "the scanner for the given contrast")),
        AcquiredFilesetSpec('coreg_ref_brain', STD_IMAGE_FORMATS,
                            desc=("A reference scan to coregister the primary "
                                  "scan to. Should be brain extracted"),
                            optional=True),
        AcquiredFilesetSpec(
            'channels', (multi_nifti_gz_format, zip_format),
            optional=True, desc=("Reconstructed complex image for each "
                                 "coil without standardisation.")),
        AcquiredFilesetSpec('header_image', dicom_format, desc=(
            "A dataset that contains correct the header information for the "
            "acquired image. Used to copy geometry over preprocessed "
            "channels"), optional=True),
        FilesetSpec('channel_mags', multi_nifti_gz_format,
                    'preprocess_channels'),
        FilesetSpec('channel_phases', multi_nifti_gz_format,
                    'preprocess_channels'),
        FilesetSpec('coreg_matrix', text_matrix_format,
                    'linear_coregistration_pipeline'),
        FilesetSpec('preproc', nifti_gz_format, 'preprocess_pipeline',
                    desc=("Performs basic preprocessing, such as realigning "
                          "image axis to a standard rotation")),
        FilesetSpec('brain', nifti_gz_format, 'brain_extraction_pipeline',
                    desc="The brain masked image"),
        FilesetSpec('brain_mask', nifti_gz_format, 'brain_extraction_pipeline',
                    desc="Mask of the brain"),
        FilesetSpec('coreg_brain', nifti_gz_format,
                    'linear_coregistration_pipeline',
                    desc="Brain coregistered to the coreg_ref_brain"),
        FilesetSpec('coreg_to_atlas', nifti_gz_format,
                    'coregister_to_atlas_pipeline'),
        FilesetSpec('coreg_to_atlas_coeff', nifti_gz_format,
                    'coregister_to_atlas_pipeline'),
        FilesetSpec('coreg_to_atlas_mat', text_matrix_format,
                    'coregister_to_atlas_pipeline'),
        FilesetSpec('coreg_to_atlas_warp', nifti_gz_format,
                    'coregister_to_atlas_pipeline'),
        FilesetSpec('coreg_to_atlas_report', gif_format,
                    'coregister_to_atlas_pipeline'),
        FilesetSpec('wm_seg', nifti_gz_format, 'segmentation_pipeline'),
        FilesetSpec('dcm_info', text_format,
                    'header_extraction_pipeline',
                    desc=("Extracts ")),
        FilesetSpec('motion_mats', motion_mats_format, 'motion_mat_pipeline'),
        FilesetSpec('qformed', nifti_gz_format, 'qform_transform_pipeline'),
        FilesetSpec('qform_mat', text_matrix_format,
                    'qform_transform_pipeline'),
        FieldSpec('tr', float, 'header_extraction_pipeline'),
        FieldSpec('echo_times', float, 'header_extraction_pipeline',
                  array=True),
        FieldSpec('voxel_sizes', float, 'header_extraction_pipeline',
                  array=True),
        FieldSpec('main_field_orient', float, 'header_extraction_pipeline',
                  array=True),
        FieldSpec('main_field_strength', float, 'header_extraction_pipeline'),
        FieldSpec('start_time', str, 'header_extraction_pipeline'),
        FieldSpec('real_duration', str, 'header_extraction_pipeline'),
        FieldSpec('total_duration', str, 'header_extraction_pipeline'),
        FieldSpec('ped', str, 'header_extraction_pipeline'),
        FieldSpec('pe_angle', str, 'header_extraction_pipeline'),
        # Templates
        AcquiredFilesetSpec('atlas', STD_IMAGE_FORMATS, frequency='per_study',
                            default=FslAtlas('MNI152_T1',
                                             resolution='atlas_resolution')),
        AcquiredFilesetSpec('atlas_brain', STD_IMAGE_FORMATS,
                            frequency='per_study',
                            default=FslAtlas('MNI152_T1',
                                             resolution='atlas_resolution',
                                             dataset='brain')),
        AcquiredFilesetSpec('atlas_mask', STD_IMAGE_FORMATS,
                            frequency='per_study',
                            default=FslAtlas('MNI152_T1',
                                             resolution='atlas_resolution',
                                             dataset='brain_mask'))]

    add_param_specs = [
        SwitchSpec('reorient_to_std', True),
        ParameterSpec('force_channel_flip', None, dtype=str, array=True,
                      desc=("Forcibly flip channel inputs during preprocess "
                            "channels to correct issues with channel recon. "
                            "The inputs are passed directly through to FSL's "
                            "swapdims (see fsl.SwapDimensions interface)")),
        SwitchSpec('bet_robust', True),
        ParameterSpec('bet_f_threshold', 0.5),
        SwitchSpec('bet_reduce_bias', False,
                   desc="Only used if not 'bet_robust'"),
        ParameterSpec('bet_g_threshold', 0.0),
        SwitchSpec('bet_method', 'fsl_bet', ('fsl_bet', 'optibet')),
        SwitchSpec('optibet_gen_report', False),
        SwitchSpec('atlas_coreg_tool', 'ants', ('fnirt', 'ants')),
        ParameterSpec('atlas_resolution', 2),  # choices=(0.5, 1, 2)),
        ParameterSpec('fnirt_intensity_model', 'global_non_linear_with_bias'),
        ParameterSpec('fnirt_subsampling', [4, 4, 2, 2, 1, 1]),
        ParameterSpec('preproc_new_dims', ('RL', 'AP', 'IS')),
        ParameterSpec('preproc_resolution', None, dtype=list),
        SwitchSpec('linear_reg_method', 'flirt', ('flirt', 'spm', 'ants'),
                   desc="The tool to use for linear registration"),
        ParameterSpec('flirt_degrees_of_freedom', 6, desc=(
            "Number of degrees of freedom used in the registration. "
            "Default is 6 -> affine transformation.")),
        ParameterSpec('flirt_cost_func', 'normmi', desc=(
            "Cost function used for the registration. Can be one of "
            "'mutualinfo', 'corratio', 'normcorr', 'normmi', 'leastsq',"
            " 'labeldiff', 'bbr'")),
        ParameterSpec('flirt_qsform', False, desc=(
            "Whether to use the QS form supplied in the input image "
            "header (the image coordinates of the FOV supplied by the "
            "scanner")),
        ParameterSpec(
            'channel_fname_regex',
            r'.*_(?P<channel>\d+)_(?P<echo>\d+)_(?P<axis>[A-Z]+)\.nii\.gz',
            desc=("The regular expression to extract channel, echo and complex"
                  " axis from the filenames of the coils channel images")),
        ParameterSpec(
            'channel_real_label', 'REAL',
            desc=("The name of the real axis extracted from the channel "
                  "filename")),
        ParameterSpec(
            'channel_imag_label', 'IMAGINARY',
            desc=("The name of the real axis extracted from the channel "
                  "filename"))]

    def preprocess_channels(self, **name_maps):
        pipeline = self.pipeline(
            'preprocess_channels',
            name_maps=name_maps,
            desc=("Convert channel signals in complex coords to polar coords "
                  "and combine"))

        if (self.input_provided('header_image') or
                self.branch('reorient_to_std') or
                self.parameter('force_channel_flip') is not None):
            # Read channel files reorient them into standard space and then
            # write back to directory
            list_channels = pipeline.add(
                'list_channels',
                ListDir(),
                inputs={
                    'directory': ('channels', multi_nifti_gz_format)})

            if self.parameter('force_channel_flip') is not None:
                force_flip = pipeline.add(
                    'flip_dims',
                    fsl.SwapDimensions(
                        new_dims=tuple(self.parameter('force_channel_flip'))),
                    connect={
                        'in_file': (list_channels, 'files')},
                    iterfield=['in_file'])
                geom_dest_file = (force_flip, 'out_file')
            else:
                geom_dest_file = (list_channels, 'files')

            if self.input_provided('header_image'):
                # If header image is provided stomp its geometry over the
                # acquired channels
                copy_geom = pipeline.add(
                    'qsm_copy_geometry',
                    fsl.CopyGeom(),
                    inputs={
                        'in_file': ('header_image', nifti_gz_format)},
                    connect={
                        'dest_file': geom_dest_file},
                    iterfield=(['dest_file']),
                    requirements=[fsl5_req])
                reorient_in_file = (copy_geom, 'out_file')
            else:
                reorient_in_file = geom_dest_file

            if self.branch('reorient_to_std'):
                reorient = pipeline.add(
                    'reorient_channel',
                    fsl.Reorient2Std(
                        output_type='NIFTI_GZ'),
                    connect={
                        'in_file': reorient_in_file},
                    iterfield=['in_file'],
                    requirements=[fsl5_req])
                copy_to_dir_in_files = (reorient, 'out_file')
            else:
                copy_to_dir_in_files = reorient_in_file

            copy_to_dir = pipeline.add(
                'copy_to_dir',
                CopyToDir(),
                connect={
                    'in_files': copy_to_dir_in_files,
                    'file_names': (list_channels, 'files')})
            to_polar_in = {'connect': {'in_dir': (copy_to_dir, 'out_dir')}}
        else:
            to_polar_in = {'inputs':
                           {'in_dir': ('channels', multi_nifti_gz_format)}}

        pipeline.add(
            'to_polar',
            ToPolarCoords(
                in_fname_re=self.parameter('channel_fname_regex'),
                real_label=self.parameter('channel_real_label'),
                imaginary_label=self.parameter('channel_imag_label')),
            outputs={
                'magnitudes_dir': ('channel_mags', multi_nifti_gz_format),
                'phases_dir': ('channel_phases', multi_nifti_gz_format)},
            **to_polar_in)

        return pipeline

    @property
    def coreg_brain_spec_name(self):
        """
        The name of the fileset after registration has been applied.
        If registration is not required, i.e. a reg_ref is not supplied
        then it is simply the 'brain' fileset.
        """
        if 'coreg_ref_brain' in self.input_names:
            name = 'coreg_brain'
        else:
            name = 'brain'
        return name

    def linear_coregistration_pipeline(self, **name_maps):
        if self.branch('linear_reg_method', 'flirt'):
            pipeline = self._flirt_pipeline(
                'linear_coreg', 'brain', 'coreg_ref_brain',
                'coreg_brain', 'coreg_matrix', name_maps)
        elif self.branch('linear_reg_method', 'ants'):
            pipeline = self._ants_linear_coreg_pipeline(
                'linear_coreg', 'brain', 'coreg_ref_brain',
                'coreg_brain', 'coreg_matrix', name_maps)
        elif self.branch('linear_reg_method', 'spm'):
            raise NotImplementedError
        else:
            self.unhandled_branch('linear_reg_method')
        return pipeline

    def qform_transform_pipeline(self, **name_maps):
        return self._qform_transform_factory(
            'qform_transform', 'brain', 'coreg_ref_brain', 'qformed',
            'qform_mat', **name_maps)

    def _flirt_pipeline(self, name, to_reg, ref, reg, matrix, name_maps):
        """
        Registers a MR scan to a refernce MR scan using FSL's FLIRT command

        Parameters
        ----------
        name : str
            Name for the generated pipeline
        to_reg : str
            Name of the FilesetSpec to register
        ref : str
            Name of the FilesetSpec to use as a reference
        reg : str
            Name of the FilesetSpec to output as registered image
        matrix : str
            Name of the FilesetSpec to output as registration matrix
        """

        pipeline = self.pipeline(
            name=name,
            name_maps=name_maps,
            desc="Registers a MR scan against a reference image using FLIRT",
            references=[fsl_cite])

        pipeline.add(
            'flirt',
            FLIRT(dof=self.parameter('flirt_degrees_of_freedom'),
                  cost=self.parameter('flirt_cost_func'),
                  cost_func=self.parameter('flirt_cost_func'),
                  output_type='NIFTI_GZ'),
            inputs={
                'in_file': (to_reg, nifti_gz_format),
                'reference': (ref, nifti_gz_format)},
            outputs={
                'out_file': (reg, nifti_gz_format),
                'out_matrix_file': (matrix, text_matrix_format)},
            requirements=[fsl5_req], wall_time=5)

        return pipeline

    def _qform_transform_factory(self, name, to_reg, ref, qformed,
                                 qformed_mat, name_maps):
        pipeline = self.pipeline(
            name=name,
            name_maps=name_maps,
            desc="Registers a MR scan against a reference image",
            references=[fsl_cite])

        pipeline.add(
            'flirt',
            FLIRT(
                uses_qform=True,
                apply_xfm=True),
            inputs={
                'in_file': (to_reg, nifti_gz_format),
                'reference': (ref, nifti_gz_format)},
            outputs={
                'out_file': (qformed, nifti_gz_format),
                'out_matrix_file': (qformed_mat, text_matrix_format)},
            requirements=[fsl5_req], wall_time=5)

        return pipeline

    def _spm_coreg_pipeline(self, **name_maps):  # @UnusedVariable
        """
        Coregisters T2 image to T1 image using SPM's
        "Register" method.

        NB: Default values come from the W2MHS toolbox
        """
        pipeline = self.pipeline(
            'registration',
            name_maps=name_maps,
            desc="Coregister T2-weighted images to T1",
            references=[spm_cite])

        pipeline.add(
            'coreg',
            Coregister(
                jobtype='estwrite',
                cost_function='nmi',
                separation=[4, 2],
                tolerance=[0.02, 0.02, 0.02, 0.001, 0.001, 0.001, 0.01, 0.01,
                           0.01, 0.001, 0.001, 0.001],
                fwhm=[7, 7],
                write_interp=4,
                write_wrap=[0, 0, 0],
                write_mask=False,
                out_prefix='r'),
            inputs={
                'target': ('t1', nifti_format),
                'source': ('t2', nifti_format)},
            outputs={
                'coregistered_source': ('t2_coreg_t1', nifti_format)},
            requirements=[spm12_req], wall_time=30)
        return pipeline

    def _ants_linear_coreg_pipeline(self, name, to_reg, ref, reg, matrix,
                                    name_maps):
        pipeline = self.pipeline(
            name=name,
            name_maps=name_maps,
            desc="Registers a MR scan against a reference image using ANTs")

        pipeline.add(
            'ANTs_linear_Reg',
            AntsRegSyn(
                num_dimensions=3,
                transformation='r',
                out_prefix='reg2hires'),
            inputs={
                'ref_file': (ref, nifti_gz_format),
                'input_file': (to_reg, nifti_gz_format)},
            outputs={
                'reg_file': (reg, nifti_gz_format),
                'regmat': (matrix, text_matrix_format)},
            wall_time=10, requirements=[ants2_req])

        return pipeline

    def brain_extraction_pipeline(self, in_file='preproc', **name_maps):
        if self.branch('bet_method', 'fsl_bet'):
            pipeline = self._fsl_bet_brain_extraction_pipeline(in_file,
                                                               **name_maps)
        elif self.branch('bet_method', 'optibet'):
            pipeline = self._optiBET_brain_extraction_pipeline(in_file,
                                                               **name_maps)
        else:
            self.unhandled_branch('bet_method')
        return pipeline

    def _fsl_bet_brain_extraction_pipeline(self, in_file, **name_maps):
        """
        Generates a whole brain mask using FSL's BET command.
        """
        pipeline = self.pipeline(
            name='brain_extraction',
            name_maps=name_maps,
            desc="Generate brain mask from mr_scan",
            references=[fsl_cite, bet_cite, bet2_cite])
        # Create mask node
        bet = pipeline.add(
            "bet",
            fsl.BET(
                mask=True,
                output_type='NIFTI_GZ',
                frac=self.parameter('bet_f_threshold'),
                vertical_gradient=self.parameter('bet_g_threshold')),
            inputs={
                'in_file': (in_file, nifti_gz_format)},
            outputs={
                'out_file': ('brain', nifti_gz_format),
                'mask_file': ('brain_mask', nifti_gz_format)},
            requirements=[fsl509_req])
        # Set either robust or reduce bias
        if self.branch('bet_robust'):
            bet.inputs.robust = True
        else:
            bet.inputs.reduce_bias = self.parameter('bet_reduce_bias')
        return pipeline

    # FIXME: With the newly implemented name-mapping functionality 'in_file'
    #        does not need to be passed in this way
    def _optiBET_brain_extraction_pipeline(self, in_file, **name_maps):
        """
        Generates a whole brain mask using a modified optiBET approach.
        """
        pipeline = self.pipeline(
            name='brain_extraction',
            name_maps=name_maps,
            desc=("Modified implementation of optiBET.sh"),
            references=[fsl_cite])

        mni_reg = pipeline.add(
            'T1_reg',
            AntsRegSyn(
                num_dimensions=3,
                transformation='s',
                out_prefix='T12MNI',
                num_threads=4),
            inputs={
                'ref_file': ('atlas', nifti_gz_format),
                'input_file': (in_file, nifti_gz_format)},
            wall_time=25, requirements=[ants2_req])

        merge_trans = pipeline.add(
            'merge_transforms',
            Merge(2),
            connect={
                'in1': (mni_reg, 'inv_warp'),
                'in2': (mni_reg, 'regmat')},
            wall_time=1)

        trans_flags = pipeline.add(
            'trans_flags',
            Merge(2,
                  in1=False,
                  in2=True),
            wall_time=1)

        apply_trans = pipeline.add(
            'ApplyTransform',
            ApplyTransforms(
                interpolation='NearestNeighbor',
                input_image_type=3),
            inputs={
                'input_image': ('atlas_mask', nifti_gz_format),
                'reference_image': (in_file, nifti_gz_format)},
            connect={
                'transforms': (merge_trans, 'out'),
                'invert_transform_flags': (trans_flags, 'out')},
            wall_time=7,
            memory=24000,
            requirements=[ants2_req])

        maths1 = pipeline.add(
            'binarize',
            fsl.ImageMaths(
                suffix='_optiBET_brain_mask',
                op_string='-bin'),
            connect={
                'in_file': (apply_trans, 'output_image')},
            outputs={
                'out_file': ('brain_mask', nifti_gz_format)},
            wall_time=5, requirements=[fsl5_req])

        maths2 = pipeline.add(
            'mask',
            fsl.ImageMaths(
                suffix='_optiBET_brain',
                op_string='-mas'),
            inputs={
                'in_file': (in_file, nifti_gz_format)},
            connect={
                'in_file2': (maths1, 'out_file')},
            outputs={
                'out_file': ('brain', nifti_gz_format)},
            wall_time=5, requirements=[fsl5_req])

        if self.branch('optibet_gen_report'):
            pipeline.add(
                'slices',
                FSLSlices(outname='optiBET_report'),
                wall_time=5,
                inputs={
                    'im1': (in_file, nifti_gz_format)},
                connect={
                    'im2': (maths2, 'out_file')},
                outputs={
                    'report': ('optiBET_report', gif_format)},
                requirements=[fsl5_req])

        return pipeline

    def coregister_to_atlas_pipeline(self, **name_maps):
        if self.branch('atlas_coreg_tool', 'fnirt'):
            pipeline = self._fsl_fnirt_to_atlas_pipeline(**name_maps)
        elif self.branch('atlas_coreg_tool', 'ants'):
            pipeline = self._ants_to_atlas_pipeline(**name_maps)
        else:
            self.unhandled_branch('atlas_coreg_tool')
        return pipeline

    # @UnusedVariable @IgnorePep8
    def _fsl_fnirt_to_atlas_pipeline(self, **name_maps):
        """
        Registers a MR scan to a refernce MR scan using FSL's nonlinear FNIRT
        command

        Parameters
        ----------
        atlas : Which atlas to use, can be one of 'mni_nl6'
        """
        pipeline = self.pipeline(
            name='coregister_to_atlas',
            name_maps=name_maps,
            desc=("Nonlinearly registers a MR scan to a standard space,"
                  "e.g. MNI-space"),
            references=[fsl_cite])

        # Basic reorientation to standard MNI space
        # FIXME: Don't think is necessary any more since preproc should be
        #        in standard orientation
        reorient = pipeline.add(
            'reorient',
            Reorient2Std(
                output_type='NIFTI_GZ'),
            inputs={
                'in_file': ('preproc', nifti_gz_format)},
            requirements=[fsl5_req])

        reorient_mask = pipeline.add(
            'reorient_mask',
            Reorient2Std(
                output_type='NIFTI_GZ'),
            inputs={
                'in_file': ('brain_mask', nifti_gz_format)},
            requirements=[fsl5_req])

        reorient_brain = pipeline.create_node(
            'reorient_brain',
            Reorient2Std(
                output_type='NIFTI_GZ'),
            inputs={
                'in_file': ('brain', nifti_gz_format)},
            requirements=[fsl5_req])

        # Affine transformation to MNI space
        flirt = pipeline.add(
            'flirt',
            interface=FLIRT(
                dof=12,
                output_type='NIFTI_GZ'),
            inputs={
                'reference': ('atlas_brain', nifti_gz_format)},
            connect={
                'in_file': (reorient_brain, 'out_file')},
            requirements=[fsl5_req],
            wall_time=5)

        # Apply mask if corresponding subsampling scheme is 1
        # (i.e. 1-to-1 resolution) otherwise don't.
        apply_mask = [int(s == 1)
                      for s in self.parameter('fnirt_subsampling')]
        # Nonlinear transformation to MNI space
        pipeline.add(
            'fnirt',
            interface=FNIRT(
                output_type='NIFTI_GZ',
                intensity_mapping_model=(
                    self.parameter('fnirt_intensity_model')
                    if self.parameter('fnirt_intensity_model') is not None else
                    'none'),
                subsampling_scheme=self.parameter('fnirt_subsampling'),
                fieldcoeff_file=True,
                in_fwhm=[8, 6, 5, 4, 3, 2],  # [8, 6, 5, 4.5, 3, 2] This threw an error because of float value @IgnorePep8,
                ref_fwhm=[8, 6, 5, 4, 2, 0],
                regularization_lambda=[300, 150, 100, 50, 40, 30],
                apply_intensity_mapping=[1, 1, 1, 1, 1, 0],
                max_nonlin_iter=[5, 5, 5, 5, 5, 10],
                apply_inmask=apply_mask,
                apply_refmask=apply_mask),
            inputs={
                'ref_file': ('atlas', nifti_gz_format),
                'refmask': ('atlas_mask', nifti_gz_format)},
            connect={
                'in_file': (reorient, 'out_file'),
                'inmask_file': (reorient_mask, 'out_file'),
                'affine_file': (flirt, 'out_matrix_file')},
            outputs={
                'warped_file': ('coreg_to_atlas', nifti_gz_format),
                'fieldcoeff_file': ('coreg_to_atlas_coeff', nifti_gz_format)},
            requirements=[fsl5_req],
            wall_time=60)
        # Set registration parameters
        # TODO: Need to work out which parameters to use
        return pipeline

    def _ants_to_atlas_pipeline(self, **name_maps):

        pipeline = self.pipeline(
            name='coregister_to_atlas',
            name_maps=name_maps,
            desc=("Nonlinearly registers a MR scan to a standard space,"
                  "e.g. MNI-space"),
            references=[fsl_cite])

        ants_reg = pipeline.add(
            'Struct2MNI_reg',
            AntsRegSyn(
                num_dimensions=3,
                transformation='s',
                out_prefix='Struct2MNI',
                num_threads=4),
            inputs={
                'input_file': (self.coreg_brain_spec_name, nifti_gz_format),
                'ref_file': ('atlas_brain', nifti_gz_format)},
            outputs={
                'reg_file': ('coreg_to_atlas', nifti_gz_format),
                'regmat': ('coreg_to_atlas_mat', text_matrix_format),
                'warp_file': ('coreg_to_atlas_warp', nifti_gz_format)},
            wall_time=25, requirements=[ants2_req])

        pipeline.add(
            'slices',
            FSLSlices(
                outname='coreg_to_atlas_report'),
            inputs={
                'im1': ('atlas', nifti_gz_format)},
            connect={
                'im2': (ants_reg, 'reg_file')},
            outputs={
                'report': ('coreg_to_atlas_report', gif_format)},
            wall_time=1, requirements=[fsl5_req])

        return pipeline

    def segmentation_pipeline(self, img_type=2, **name_maps):
        pipeline = self.pipeline(
            name='FAST_segmentation',
            name_maps=name_maps,
            inputs=[FilesetSpec('brain', nifti_gz_format)],
            outputs=[FilesetSpec('wm_seg', nifti_gz_format)],
            desc="White matter segmentation of the reference image",
            references=[fsl_cite])

        fast = pipeline.add(
            'fast',
            fsl.FAST(
                img_type=img_type,
                segments=True,
                out_basename='Reference_segmentation'),
            inputs={
                'in_files': ('brain', nifti_gz_format)},
            requirements=[fsl509_req]),

        # Determine output field of split to use
        if img_type == 1:
            split_output = 'out3'
        elif img_type == 2:
            split_output = 'out2'
        else:
            raise ArcanaUsageError(
                "'img_type' parameter can either be 1 or 2 (not {})"
                .format(img_type))

        pipeline.add(
            'split',
            Split(
                splits=[1, 1, 1],
                squeeze=True),
            connect={
                'inlist': (fast, 'tissue_class_files')},
            outputs={
                split_output: ('wm_seg', nifti_gz_format)})

        return pipeline

    def preprocess_pipeline(self, **name_maps):
        """
        Performs basic preprocessing, such as swapping dimensions into
        standard orientation and resampling (if required)

        Parameters
        -------
        new_dims : tuple(str)[3]
            A 3-tuple with the new orientation of the image (see FSL
            swap dim)
        resolution : list(float)[3] | None
            New resolution of the image. If None no resampling is
            performed
        """
        pipeline = self.pipeline(
            name='preprocess_pipeline',
            name_maps=name_maps,
            desc=("Dimensions swapping to ensure that all the images "
                  "have the same orientations."),
            references=[fsl_cite])

        if (self.branch('reorient_to_std') or
                self.parameter('preproc_resolution') is not None):
            if self.branch('reorient_to_std'):
                swap = pipeline.add(
                    'fslreorient2std',
                    fsl.utils.Reorient2Std(),
                    inputs={
                        'in_file': ('magnitude', nifti_gz_format)},
                    requirements=[fsl509_req])
    #         swap.inputs.new_dims = self.parameter('preproc_new_dims')

            if self.parameter('preproc_resolution') is not None:
                resample = pipeline.add(
                    "resample",
                    MRResize(
                        voxel=self.parameter('preproc_resolution')),
                    connect={'in_file': (swap, 'out_file')},
                    requirements=[mrtrix3_req])
                pipeline.connect_output('preproc', resample, 'out_file',
                                        nifti_gz_format)
            else:
                pipeline.connect_output('preproc', swap, 'out_file',
                                        nifti_gz_format)
        else:
            # Don't actually do any processing just copy magnitude image to
            # preproc
            pipeline.add(
                'identity',
                IdentityInterface(
                    ['file']),
                inputs={
                    'file': ('magnitude', nifti_gz_format)},
                outputs={
                    'file': ('preproc', nifti_gz_format)})

        return pipeline

    def header_extraction_pipeline(self, **name_maps):
        if self.input_provided('header_image'):
            dcm_in_name = 'header_image'
        else:
            dcm_in_name = 'magnitude'
        if self.input(dcm_in_name).format != dicom_format:
            raise ArcanaUsageError(
                "Can only extract header info if 'magnitude' fileset "
                "is provided in DICOM format ({})".format(
                    self.input('magnitude').format))

        pipeline = self.pipeline(
            name='header_extraction',
            name_maps=name_maps,
            desc=("Pipeline to extract the most important scan "
                  "information from the image header"),
            references=[])

        pipeline.add(
            'hd_info_extraction',
            DicomHeaderInfoExtraction(
                multivol=False),
            inputs={
                'dicom_folder': (dcm_in_name, dicom_format)},
            outputs={
                'tr': ('tr', float),
                'start_time': ('start_time', str),
                'total_duration': ('total_duration', str),
                'real_duration': ('real_duration', str),
                'ped': ('ped', str),
                'pe_angle': ('pe_angle', str),
                'dcm_info': ('dcm_info', text_format),
                'echo_times': ('echo_times', float),
                'voxel_sizes': ('voxel_sizes', float),
                'B0': ('main_field_strength', float),
                'H': ('main_field_orient', float)})

        return pipeline

    def motion_mat_pipeline(self, **name_maps):

        pipeline = self.pipeline(
            name='motion_mat_calculation',
            name_maps=name_maps,
            desc=("Motion matrices calculation"),
            references=[fsl_cite])

        mm = pipeline.add(
            'motion_mats',
            MotionMatCalculation(),
            outputs={
                'motion_mats': ('motion_mats', motion_mats_format)})
        if not self.spec('coreg_matrix').derivable:
            logger.info("Cannot derive 'coreg_matrix' for {} required for "
                        "motion matrix calculation, assuming that it "
                        "is the reference study".format(self))
            mm.inputs.reference = True
            pipeline.connect_input('magnitude', mm, 'dummy_input')
        else:
            pipeline.connect_input('coreg_matrix', mm, 'reg_mat',
                                   text_matrix_format)
            pipeline.connect_input('qform_mat', mm, 'qform_mat',
                                   text_matrix_format)
            if 'align_mats' in self.data_spec_names():
                pipeline.connect_input('align_mats', mm, 'align_mats',
                                       directory_format)
        return pipeline
