from nipype.interfaces import fsl
from nipype.interfaces.spm.preprocess import Coregister
from banana.citation import spm_cite
from banana.file_format import (
    nifti_format, motion_mats_format, nifti_gz_format,
    multi_nifti_gz_format, zip_format, STD_IMAGE_FORMATS)
from arcana.data import FilesetSpec, FieldSpec, InputFilesetSpec
from banana.study import Study, StudyMetaClass
from banana.citation import fsl_cite, bet_cite, bet2_cite, ants_cite
from banana.file_format import (
    dicom_format, gif_format, nifti_gz_x_format)
from nipype.interfaces.utility import IdentityInterface
from banana.requirement import fsl_req, mrtrix_req, ants_req, spm_req, c3d_req
from nipype.interfaces.fsl import (FLIRT, FNIRT, Reorient2Std)
from arcana.exceptions import (
    ArcanaOutputNotProducedException, ArcanaMissingDataException)
from banana.interfaces.mrtrix.transform import MRResize
from banana.interfaces.custom.dicom import (
    DicomHeaderInfoExtraction, NiftixHeaderInfoExtraction)
from nipype.interfaces.utility import Merge
from nipype.interfaces import ants
from banana.interfaces.fsl import FSLSlices
from banana.file_format import text_matrix_format
import logging
from banana.interfaces.ants import AntsRegSyn
from banana.interfaces.custom.coils import ToPolarCoords
from arcana.utils.interfaces import ListDir, CopyToDir
from nipype.interfaces.ants.resampling import ApplyTransforms
from nipype.interfaces.fsl.preprocess import ApplyXFM
from banana.interfaces.c3d import ANTs2FSLMatrixConversion
from arcana import ParamSpec, SwitchSpec
from banana.interfaces.custom.motion_correction import (
    MotionMatCalculation)
from banana.exceptions import BananaUsageError
from banana.reference import FslReferenceData

logger = logging.getLogger('arcana')


class MriStudy(Study, metaclass=StudyMetaClass):

    add_data_specs = [
        InputFilesetSpec('magnitude', STD_IMAGE_FORMATS,
                         desc=("Typically the primary scan acquired from "
                               "the scanner for the given contrast")),
        InputFilesetSpec(
            'coreg_ref', STD_IMAGE_FORMATS,
            desc=("A reference scan to coregister the primary scan to. Should "
                  "not be brain extracted"),
            optional=True),
        InputFilesetSpec(
            'coreg_ref_brain', STD_IMAGE_FORMATS,
            desc=("A brain-extracted reference scan to coregister a brain-"
                  "extracted scan to. Note that the output of the "
                  "registration brain_coreg can also be derived by brain "
                  "extracting the output of coregistration performed "
                  "before brain extraction if 'coreg_ref' is provided"),
            optional=True),
        InputFilesetSpec(
            'channels', (multi_nifti_gz_format, zip_format),
            optional=True, desc=("Reconstructed complex image for each "
                                 "coil without standardisation.")),
        InputFilesetSpec('header_image', dicom_format, desc=(
            "A dataset that contains correct the header information for the "
            "acquired image. Used to copy geometry over preprocessed "
            "channels"), optional=True),
        FilesetSpec('mag_preproc', nifti_gz_format, 'prepare_pipeline',
                    desc=("Magnitude after basic preprocessing, such as "
                          "realigning image axis to a standard rotation")),
        FilesetSpec('mag_channels', multi_nifti_gz_format,
                    'preprocess_channels_pipeline'),
        FilesetSpec('phase_channels', multi_nifti_gz_format,
                    'preprocess_channels_pipeline'),
        FilesetSpec('brain', nifti_gz_format, 'brain_extraction_pipeline',
                    desc="The brain masked image"),
        FilesetSpec('brain_mask', nifti_gz_format, 'brain_extraction_pipeline',
                    desc="Mask of the brain"),
        FilesetSpec('mag_coreg', nifti_gz_format, 'coreg_pipeline',
                    desc="Head image coregistered to 'coreg_ref'"),
        FilesetSpec('brain_coreg', nifti_gz_format,
                    'brain_coreg_pipeline',
                    desc=("Either brain-extracted image coregistered to "
                          "'coreg_ref_brain' or a brain extraction of a "
                          "coregistered (incl. skull) image")),
        FilesetSpec('brain_mask_coreg', nifti_gz_format,
                    'brain_coreg_pipeline',
                    desc=("Either brain-extracted image coregistered to "
                          "'coreg_ref_brain' or a brain extraction of a "
                          "coregistered (incl. skull) image")),
        FilesetSpec('coreg_ants_mat', text_matrix_format,
                    'coreg_ants_mat_pipeline'),
        FilesetSpec('coreg_fsl_mat', text_matrix_format,
                    'coreg_fsl_mat_pipeline'),
        FilesetSpec('mag_coreg_to_tmpl', nifti_gz_format,
                    'coreg_to_tmpl_pipeline'),
        FilesetSpec('coreg_to_tmpl_fsl_coeff', nifti_gz_format,
                    'coreg_to_tmpl_pipeline'),
        FilesetSpec('coreg_to_tmpl_fsl_report', gif_format,
                    'coreg_to_tmpl_pipeline'),
        FilesetSpec('coreg_to_tmpl_ants_mat', text_matrix_format,
                    'coreg_to_tmpl_pipeline'),
        FilesetSpec('coreg_to_tmpl_ants_warp', nifti_gz_format,
                    'coreg_to_tmpl_pipeline'),
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
        FieldSpec('start_time', float, 'header_extraction_pipeline'),
        FieldSpec('real_duration', float, 'header_extraction_pipeline'),
        FieldSpec('total_duration', float, 'header_extraction_pipeline'),
        FieldSpec('ped', str, 'header_extraction_pipeline'),
        FieldSpec('pe_angle', float, 'header_extraction_pipeline'),
        # Templates
        InputFilesetSpec('template', STD_IMAGE_FORMATS, frequency='per_study',
                         default=FslReferenceData(
                             'MNI152_T1',
                             format=nifti_gz_format,
                             resolution='mni_template_resolution')),
        InputFilesetSpec('template_brain', STD_IMAGE_FORMATS,
                         frequency='per_study',
                         default=FslReferenceData(
                             'MNI152_T1',
                             format=nifti_gz_format,
                             resolution='mni_template_resolution',
                             dataset='brain')),
        InputFilesetSpec('template_mask', STD_IMAGE_FORMATS,
                         frequency='per_study',
                         default=FslReferenceData(
                             'MNI152_T1',
                             format=nifti_gz_format,
                             resolution='mni_template_resolution',
                             dataset='brain_mask'))]

    add_param_specs = [
        SwitchSpec('resample_coreg_ref', False,
                   desc=("Whether to resample the coregistration reference "
                         "image to the resolution of the moving image")),
        SwitchSpec('reorient_to_std', True),
        ParamSpec('force_channel_flip', None, dtype=str, array=True,
                      desc=("Forcibly flip channel inputs during preprocess "
                            "channels to correct issues with channel recon. "
                            "The inputs are passed directly through to FSL's "
                            "swapdims (see fsl.SwapDimensions interface)")),
        SwitchSpec('bet_robust', True),
        ParamSpec('bet_f_threshold', 0.5),
        SwitchSpec('bet_reduce_bias', False,
                   desc="Only used if not 'bet_robust'"),
        ParamSpec('bet_g_threshold', 0.0),
        SwitchSpec('bet_method', 'fsl_bet', ('fsl_bet', 'optibet')),
        SwitchSpec('optibet_gen_report', False),
        SwitchSpec('coreg_to_tmpl_method', 'ants', ('fnirt', 'ants')),
        ParamSpec('mni_template_resolution', None, choices=(0.5, 1, 2),
                  dtype=int),
        ParamSpec('fnirt_intensity_model', 'global_non_linear_with_bias'),
        ParamSpec('fnirt_subsampling', [4, 4, 2, 2, 1, 1]),
        ParamSpec('reoriented_dims', ('RL', 'AP', 'IS')),
        ParamSpec('resampled_resolution', None, dtype=list),
        SwitchSpec('coreg_method', 'ants', ('ants', 'flirt', 'spm'),
                   desc="The tool to use for linear registration"),
        ParamSpec('flirt_degrees_of_freedom', 6, desc=(
            "Number of degrees of freedom used in the registration. "
            "Default is 6 -> affine transformation.")),
        ParamSpec('flirt_cost_func', 'normmi', desc=(
            "Cost function used for the registration. Can be one of "
            "'mutualinfo', 'corratio', 'normcorr', 'normmi', 'leastsq',"
            " 'labeldiff', 'bbr'")),
        ParamSpec('flirt_qsform', False, desc=(
            "Whether to use the QS form supplied in the input image "
            "header (the image coordinates of the FOV supplied by the "
            "scanner")),
        ParamSpec(
            'channel_fname_regex',
            r'.*_(?P<channel>\d+)_(?P<echo>\d+)_(?P<axis>[A-Z]+)\.nii\.gz',
            desc=("The regular expression to extract channel, echo and complex"
                  " axis from the filenames of the coils channel images")),
        ParamSpec(
            'channel_real_label', 'REAL',
            desc=("The name of the real axis extracted from the channel "
                  "filename")),
        ParamSpec(
            'channel_imag_label', 'IMAGINARY',
            desc=("The name of the real axis extracted from the channel "
                  "filename"))]

    @property
    def mni_template_resolution(self):
        if self.parameter('mni_template_resolution') is not None:
            res = self.parameter('mni_template_resolution')
        else:
            raise ArcanaMissingDataException(
                "Automatic detection of dataset resolution is not implemented "
                "yet, please specify resolution of default MNI templates "
                "manually via 'mni_template_resolution' parameter")
        return res

    @property
    def is_coregistered(self):
        return self.provided('coreg_ref') or self.provided('coreg_ref_brain')

    @property
    def header_image_spec_name(self):
        if self.provided('header_image'):
            hdr_name = 'header_image'
        else:
            hdr_name = 'magnitude'
        return hdr_name

    @property
    def brain_spec_name(self):
        """
        The name of the brain extracted image after registration has been
        applied if registration is specified by supplying 'coreg_ref' or
        'coreg_ref_brain' optional inputs.
        """
        if self.is_coregistered:
            name = 'brain_coreg'
        else:
            name = 'brain'
        return name

    @property
    def brain_mask_spec_name(self):
        if self.is_coregistered:
            brain_mask = 'brain_mask_coreg'
        else:
            brain_mask = 'brain_mask'
        return brain_mask

    def preprocess_channels_pipeline(self, **name_maps):
        pipeline = self.new_pipeline(
            'preprocess_channels',
            name_maps=name_maps,
            desc=("Convert channel signals in complex coords to polar coords "
                  "and combine"))

        if (self.provided('header_image') or
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
                    inputs={
                        'in_file': (list_channels, 'files')},
                    iterfield=['in_file'])
                geom_dest_file = (force_flip, 'out_file')
            else:
                geom_dest_file = (list_channels, 'files')

            if self.provided('header_image'):
                # If header image is provided stomp its geometry over the
                # acquired channels
                copy_geom = pipeline.add(
                    'qsm_copy_geometry',
                    fsl.CopyGeom(
                        output_type='NIFTI_GZ'),
                    inputs={
                        'in_file': ('header_image', nifti_gz_format),
                        'dest_file': geom_dest_file},
                    iterfield=(['dest_file']),
                    requirements=[fsl_req.v('5.0.8')])
                reorient_in_file = (copy_geom, 'out_file')
            else:
                reorient_in_file = geom_dest_file

            if self.branch('reorient_to_std'):
                reorient = pipeline.add(
                    'reorient_channel',
                    fsl.Reorient2Std(
                        output_type='NIFTI_GZ'),
                    inputs={
                        'in_file': reorient_in_file},
                    iterfield=['in_file'],
                    requirements=[fsl_req.v('5.0.8')])
                copy_to_dir_in_files = (reorient, 'out_file')
            else:
                copy_to_dir_in_files = reorient_in_file

            copy_to_dir = pipeline.add(
                'copy_to_dir',
                CopyToDir(),
                inputs={
                    'in_files': copy_to_dir_in_files,
                    'file_names': (list_channels, 'files')})
            to_polar_in_dir = (copy_to_dir, 'out_dir')
        else:
            to_polar_in_dir = ('channels', multi_nifti_gz_format)

        pipeline.add(
            'to_polar',
            ToPolarCoords(
                in_fname_re=self.parameter('channel_fname_regex'),
                real_label=self.parameter('channel_real_label'),
                imaginary_label=self.parameter('channel_imag_label')),
            inputs={
                'in_dir': to_polar_in_dir},
            outputs={
                'mag_channels': ('magnitudes_dir', multi_nifti_gz_format),
                'phase_channels': ('phases_dir', multi_nifti_gz_format)})

        return pipeline

    def coreg_pipeline(self, **name_maps):
        if self.branch('coreg_method', 'flirt'):
            pipeline = self._flirt_linear_coreg_pipeline(**name_maps)
        elif self.branch('coreg_method', 'ants'):
            pipeline = self._ants_linear_coreg_pipeline(**name_maps)
        elif self.branch('coreg_method', 'spm'):
            pipeline = self._spm_linear_coreg_pipeline(**name_maps)
        else:
            self.unhandled_branch('coreg_method')
        if not self.provided(pipeline.map_input('coreg_ref')):
            raise ArcanaOutputNotProducedException(
                "Cannot co-register {} as reference image "
                "'{}' has not been provided".format(
                    pipeline.map_input('coreg_ref')))
        return pipeline

    def brain_extraction_pipeline(self, **name_maps):
        if self.branch('bet_method', 'fsl_bet'):
            pipeline = self._bet_brain_extraction_pipeline(**name_maps)
        elif self.branch('bet_method', 'optibet'):
            pipeline = self._optiBET_brain_extraction_pipeline(**name_maps)
        else:
            self.unhandled_branch('bet_method')
        return pipeline

    def brain_coreg_pipeline(self, **name_maps):
        """
        Coregistered + brain-extracted images can be derived in 2-ways. If an
        explicit brain-extracted reference is provided to
        'coreg_ref_brain' then that is used to coregister a brain extracted
        image against. Alternatively, if only a skull-included reference is
        provided then the registration is performed with skulls-included and
        then brain extraction is performed after
        """
        if self.provided('coreg_ref_brain'):
            # If a reference brain extracted image is provided we coregister
            # the brain extracted image to that
            pipeline = self.coreg_pipeline(
                name='brain_coreg',
                name_maps=dict(
                    input_map={
                        'mag_preproc': 'brain',
                        'coreg_ref': 'coreg_ref_brain'},
                    output_map={
                        'mag_coreg': 'brain_coreg'},
                    name_maps=name_maps))

            # Apply coregistration transform to brain mask
            if self.branch('coreg_method', 'flirt'):
                pipeline.add(
                    'mask_transform',
                    ApplyXFM(
                        output_type='NIFTI_GZ',
                        apply_xfm=True),
                    inputs={
                        'in_matrix_file': (pipeline.node('flirt'),
                                           'out_matrix_file'),
                        'in_file': ('brain_mask', nifti_gz_format),
                        'reference': ('coreg_ref_brain', nifti_gz_format)},
                    outputs={
                        'brain_mask_coreg': ('out_file', nifti_gz_format)},
                    requirements=[fsl_req.v('5.0.10')],
                    wall_time=10)

            elif self.branch('coreg_method', 'ants'):
                # Convert ANTs transform matrix to FSL format if we have used
                # Ants registration so we can apply the transform using
                # ApplyXFM
                pipeline.add(
                    'mask_transform',
                    ants.resampling.ApplyTransforms(
                        interpolation='Linear',
                        input_image_type=3,
                        invert_transform_flags=[True, True, False]),
                    inputs={
                        'input_image': ('brain_mask', nifti_gz_format),
                        'reference_image': ('coreg_ref_brain',
                                            nifti_gz_format),
                        'transforms': (pipeline.node('ants_reg'),
                                       'forward_transforms')},
                    requirements=[ants_req.v('1.9')], mem_gb=16,
                    wall_time=30)
            else:
                self.unhandled_branch('coreg_method')

        elif self.provided('coreg_ref'):
            # If coreg_ref is provided then we co-register the non-brain
            # extracted images and then brain extract the co-registered image
            pipeline = self.brain_extraction_pipeline(
                name='bet_coreg',
                input_map={'mag_preproc': 'mag_coreg'},
                output_map={'brain': 'brain_coreg',
                            'brain_mask': 'brain_mask_coreg'},
                name_maps=name_maps)
        else:
            raise BananaUsageError(
                "Either 'coreg_ref' or 'coreg_ref_brain' needs to be provided "
                "in order to derive brain_coreg or brain_mask_coreg")
        return pipeline

    def _coreg_mat_pipeline(self, **name_maps):
        if self.provided('coreg_ref_brain'):
            pipeline = self.brain_coreg_pipeline(**name_maps)
        elif self.provided('coreg_ref'):
            pipeline = self.coreg_pipeline(**name_maps)
        else:
            raise ArcanaOutputNotProducedException(
                "'coregistration matrices can only be derived if 'coreg_ref' "
                "or 'coreg_ref_brain' is provided to {}".format(self))
        return pipeline

    def coreg_ants_mat_pipeline(self, **name_maps):
        if self.branch('coreg_method', 'ants'):
            pipeline = self._coreg_mat_pipeline(**name_maps)
        else:
            # Run the coreg_mat pipeline only to generate the ANTs transform
            # and mapping the typical outputs to None so they don't override
            # the other settings
            pipeline = self._coreg_mat_pipeline(
                output_maps={
                    'mag_preproc': None,
                    'brain_coreg': None,
                    'brain_mask_coreg': None},
                name_maps=name_maps)
        return pipeline

    def coreg_fsl_mat_pipeline(self, **name_maps):
        if self.branch('coreg_method', 'flirt'):
            pipeline = self._coreg_mat_pipeline(**name_maps)
        elif self.branch('coreg_method', 'ants'):
            # Convert ANTS transform to FSL transform
            pipeline = self.new_pipeline(
                name='convert_ants_to_fsl_coreg_mat',
                name_maps=name_maps)

            if self.provided('coreg_ref'):
                source = 'mag_preproc'
                ref = 'coreg_ref'
            elif self.provided('coreg_ref_brain'):
                source = 'brain'
                ref = 'coreg_ref_brain'
            else:
                raise BananaUsageError(
                    "Either 'coreg_ref' or 'coreg_ref_brain' needs to be "
                    "provided in order to derive brain_coreg or brain_coreg_"
                    "mask")

            pipeline.add(
                'transform_conv',
                ANTs2FSLMatrixConversion(
                    ras2fsl=True),
                inputs={
                    'itk_file': ('coreg_ants_mat', text_matrix_format),
                    'source_file': (source, nifti_gz_format),
                    'reference_file': (ref, nifti_gz_format)},
                outputs={
                    'coreg_fsl_mat': ('fsl_matrix', text_matrix_format)},
                requirements=[c3d_req.v('1.0')])
        else:
            self.unhandled_branch('coreg_method')

        return pipeline

    def coreg_to_tmpl_pipeline(self, **name_maps):
        if self.branch('coreg_to_tmpl_method', 'fnirt'):
            pipeline = self._fnirt_to_tmpl_pipeline(**name_maps)
        elif self.branch('coreg_to_tmpl_method', 'ants'):
            pipeline = self._ants_to_tmpl_pipeline(**name_maps)
        else:
            self.unhandled_branch('coreg_to_tmpl_method')
        return pipeline

    def _flirt_linear_coreg_pipeline(self, **name_maps):
        """
        Registers a MR scan to a refernce MR scan using FSL's FLIRT command
        """

        pipeline = self.new_pipeline(
            name='linear_coreg',
            name_maps=name_maps,
            desc="Registers a MR scan against a reference image using FLIRT",
            citations=[fsl_cite])

        pipeline.add(
            'flirt',
            FLIRT(dof=self.parameter('flirt_degrees_of_freedom'),
                  cost=self.parameter('flirt_cost_func'),
                  cost_func=self.parameter('flirt_cost_func'),
                  output_type='NIFTI_GZ'),
            inputs={
                'in_file': ('mag_preproc', nifti_gz_format),
                'reference': ('coreg_ref', nifti_gz_format)},
            outputs={
                'mag_coreg': ('out_file', nifti_gz_format),
                'coreg_fsl_mat': ('out_matrix_file', text_matrix_format)},
            requirements=[fsl_req.v('5.0.8')],
            wall_time=5)

        return pipeline

    def _ants_linear_coreg_pipeline(self, **name_maps):
        """
        Registers a MR scan to a refernce MR scan using ANTS's linear_reg
        command
        """

        pipeline = self.new_pipeline(
            name='linear_coreg',
            name_maps=name_maps,
            desc="Registers a MR scan against a reference image using ANTs",
            citations=[ants_cite])

        pipeline.add(
            'ANTs_linear_Reg',
            AntsRegSyn(
                num_dimensions=3,
                transformation='r'),
            inputs={
                'ref_file': ('coreg_ref', nifti_gz_format),
                'input_file': ('mag_preproc', nifti_gz_format)},
            outputs={
                'mag_coreg': ('reg_file', nifti_gz_format),
                'coreg_ants_mat': ('regmat', text_matrix_format)},
            wall_time=10,
            requirements=[ants_req.v('2.0')])


#         ants_reg = pipeline.add(
#             'ants_reg',
#             ants.Registration(
#                 dimension=3,
#                 collapse_output_transforms=True,
#                 float=False,
#                 interpolation='Linear',
#                 use_histogram_matching=False,
#                 winsorize_upper_quantile=0.995,
#                 winsorize_lower_quantile=0.005,
#                 verbose=True,
#                 transforms=['Rigid'],
#                 transform_parameters=[(0.1,)],
#                 metric=['MI'],
#                 metric_weight=[1],
#                 radius_or_number_of_bins=[32],
#                 sampling_strategy=['Regular'],
#                 sampling_percentage=[0.25],
#                 number_of_iterations=[[1000, 500, 250, 100]],
#                 convergence_threshold=[1e-6],
#                 convergence_window_size=[10],
#                 shrink_factors=[[8, 4, 2, 1]],
#                 smoothing_sigmas=[[3, 2, 1, 0]],
#                 output_warped_image=True),
#             inputs={
#                 'fixed_image': ('coreg_ref', nifti_gz_format),
#                 'moving_image': ('mag_preproc', nifti_gz_format)},
#             outputs={
#                 'mag_coreg': ('warped_image', nifti_gz_format)},
#             wall_time=10,
#             requirements=[ants_req.v('2.0')])
#
#         pipeline.add(
#             'select',
#             SelectOne(
#                 index=0),
#             inputs={
#                 'inlist': (ants_reg, 'forward_transforms')},
#             outputs={
#                 'coreg_ants_mat': ('out', text_matrix_format)})

        return pipeline

    def _spm_linear_coreg_pipeline(self, **name_maps):  # @UnusedVariable
        """
        Coregisters T2 image to T1 image using SPM's "Register" method.

        NB: Default values come from the W2MHS toolbox
        """
        pipeline = self.new_pipeline(
            'linear_coreg',
            name_maps=name_maps,
            desc="Coregister T2-weighted images to T1",
            citations=[spm_cite])

        pipeline.add(
            'mag_coreg',
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
                'target': ('coreg_ref', nifti_format),
                'source': ('mag_preproc', nifti_format)},
            outputs={
                'mag_coreg': ('coregistered_source', nifti_format)},
            requirements=[spm_req.v(12)],
            wall_time=30)
        return pipeline

    def qform_transform_pipeline(self, **name_maps):
        pipeline = self.new_pipeline(
            name='qform_transform',
            name_maps=name_maps,
            desc="Registers a MR scan against a reference image",
            citations=[fsl_cite])

        if self.provided('coreg_ref'):
            in_file = 'mag_preproc'
            reference = 'coreg_ref'
        elif self.provided('coreg_ref_brain'):
            in_file = 'brain'
            reference = 'coreg_ref_brain'
        else:
            raise BananaUsageError(
                "'coreg_ref' or 'coreg_ref_brain' need to be provided to "
                "study in order to run qform_transform")

        pipeline.add(
            'flirt',
            FLIRT(
                uses_qform=True,
                apply_xfm=True,
                output_type='NIFTI_GZ'),
            inputs={
                'in_file': (in_file, nifti_gz_format),
                'reference': (reference, nifti_gz_format)},
            outputs={
                'qformed': ('out_file', nifti_gz_format),
                'qform_mat': ('out_matrix_file', text_matrix_format)},
            requirements=[fsl_req.v('5.0.8')],
            wall_time=5)

        return pipeline

    def _bet_brain_extraction_pipeline(self, **name_maps):
        """
        Generates a whole brain mask using FSL's BET command.
        """
        pipeline = self.new_pipeline(
            name='brain_extraction',
            name_maps=name_maps,
            desc="Generate brain mask from mr_scan",
            citations=[fsl_cite, bet_cite, bet2_cite])
        # Create mask node
        bet = pipeline.add(
            "bet",
            fsl.BET(
                mask=True,
                output_type='NIFTI_GZ',
                frac=self.parameter('bet_f_threshold'),
                vertical_gradient=self.parameter('bet_g_threshold')),
            inputs={
                'in_file': ('mag_preproc', nifti_gz_format)},
            outputs={
                'brain': ('out_file', nifti_gz_format),
                'brain_mask': ('mask_file', nifti_gz_format)},
            requirements=[fsl_req.v('5.0.9')])
        # Set either robust or reduce bias
        if self.branch('bet_robust'):
            bet.inputs.robust = True
        else:
            bet.inputs.reduce_bias = self.parameter('bet_reduce_bias')
        return pipeline

    def _optiBET_brain_extraction_pipeline(self, **name_maps):
        """
        Generates a whole brain mask using a modified optiBET approach.
        """
        pipeline = self.new_pipeline(
            name='brain_extraction',
            name_maps=name_maps,
            desc=("Modified implementation of optiBET.sh"),
            citations=[fsl_cite])

        mni_reg = pipeline.add(
            'T1_reg',
            AntsRegSyn(
                num_dimensions=3,
                transformation='s',
                out_prefix='T12MNI',
                num_threads=4),
            inputs={
                'ref_file': ('template', nifti_gz_format),
                'input_file': ('mag_preproc', nifti_gz_format)},
            wall_time=25,
            requirements=[ants_req.v('2.0')])

        merge_trans = pipeline.add(
            'merge_transforms',
            Merge(2),
            inputs={
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
                'input_image': ('template_mask', nifti_gz_format),
                'reference_image': ('mag_preproc', nifti_gz_format),
                'transforms': (merge_trans, 'out'),
                'invert_transform_flags': (trans_flags, 'out')},
            wall_time=7,
            mem_gb=24,
            requirements=[ants_req.v('2.0')])

        maths1 = pipeline.add(
            'binarize',
            fsl.ImageMaths(
                suffix='_optiBET_brain_mask',
                op_string='-bin',
                output_type='NIFTI_GZ'),
            inputs={
                'in_file': (apply_trans, 'output_image')},
            outputs={
                'brain_mask': ('out_file', nifti_gz_format)},
            wall_time=5,
            requirements=[fsl_req.v('5.0.8')])

        maths2 = pipeline.add(
            'mask',
            fsl.ImageMaths(
                suffix='_optiBET_brain',
                op_string='-mas',
                output_type='NIFTI_GZ'),
            inputs={
                'in_file': ('mag_preproc', nifti_gz_format),
                'in_file2': (maths1, 'out_file')},
            outputs={
                'brain': ('out_file', nifti_gz_format)},
            wall_time=5,
            requirements=[fsl_req.v('5.0.8')])

        if self.branch('optibet_gen_report'):
            pipeline.add(
                'slices',
                FSLSlices(
                    outname='optiBET_report',
                    output_type='NIFTI_GZ'),
                wall_time=5,
                inputs={
                    'im1': ('mag_preproc', nifti_gz_format),
                    'im2': (maths2, 'out_file')},
                outputs={
                    'optiBET_report': ('report', gif_format)},
                requirements=[fsl_req.v('5.0.8')])

        return pipeline

    # @UnusedVariable @IgnorePep8
    def _fnirt_to_tmpl_pipeline(self, **name_maps):
        """
        Registers a MR scan to a refernce MR scan using FSL's nonlinear FNIRT
        command

        Parameters
        ----------
        template : Which template to use, can be one of 'mni_nl6'
        """
        pipeline = self.new_pipeline(
            name='mag_coreg_to_tmpl',
            name_maps=name_maps,
            desc=("Nonlinearly registers a MR scan to a standard space,"
                  "e.g. MNI-space"),
            citations=[fsl_cite])

        # Basic reorientation to standard MNI space
        reorient = pipeline.add(
            'reorient',
            Reorient2Std(
                output_type='NIFTI_GZ'),
            inputs={
                'in_file': ('mag_preproc', nifti_gz_format)},
            requirements=[fsl_req.v('5.0.8')])

        reorient_mask = pipeline.add(
            'reorient_mask',
            Reorient2Std(
                output_type='NIFTI_GZ'),
            inputs={
                'in_file': ('brain_mask', nifti_gz_format)},
            requirements=[fsl_req.v('5.0.8')])

        reorient_brain = pipeline.add(
            'reorient_brain',
            Reorient2Std(
                output_type='NIFTI_GZ'),
            inputs={
                'in_file': ('brain', nifti_gz_format)},
            requirements=[fsl_req.v('5.0.8')])

        # Affine transformation to MNI space
        flirt = pipeline.add(
            'flirt',
            interface=FLIRT(
                dof=12,
                output_type='NIFTI_GZ'),
            inputs={
                'reference': ('template_brain', nifti_gz_format),
                'in_file': (reorient_brain, 'out_file')},
            requirements=[fsl_req.v('5.0.8')],
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
                'ref_file': ('template', nifti_gz_format),
                'refmask': ('template_mask', nifti_gz_format),
                'in_file': (reorient, 'out_file'),
                'inmask_file': (reorient_mask, 'out_file'),
                'affine_file': (flirt, 'out_matrix_file')},
            outputs={
                'mag_coreg_to_tmpl': ('warped_file', nifti_gz_format),
                'coreg_to_tmpl_fsl_coeff': ('fieldcoeff_file',
                                             nifti_gz_format)},
            requirements=[fsl_req.v('5.0.8')],
            wall_time=60)
        # Set registration parameters
        # TODO: Need to work out which parameters to use
        return pipeline

    def _ants_to_tmpl_pipeline(self, **name_maps):

        pipeline = self.new_pipeline(
            name='mag_coreg_to_tmpl',
            name_maps=name_maps,
            desc=("Nonlinearly registers a MR scan to a standard space,"
                  "e.g. MNI-space"),
            citations=[fsl_cite])

        pipeline.add(
            'Struct2MNI_reg',
            AntsRegSyn(
                num_dimensions=3,
                transformation='s',
                num_threads=4),
            inputs={
                'input_file': (self.brain_spec_name, nifti_gz_format),
                'ref_file': ('template_brain', nifti_gz_format)},
            outputs={
                'mag_coreg_to_tmpl': ('reg_file', nifti_gz_format),
                'coreg_to_tmpl_ants_mat': ('regmat', text_matrix_format),
                'coreg_to_tmpl_ants_warp': ('warp_file', nifti_gz_format)},
            wall_time=25,
            requirements=[ants_req.v('2.0')])

#         ants_reg = pipeline.add(
#             'ants_reg',
#             ants.Registration(
#                 dimension=3,
#                 collapse_output_transforms=True,
#                 float=False,
#                 interpolation='Linear',
#                 use_histogram_matching=False,
#                 winsorize_upper_quantile=0.995,
#                 winsorize_lower_quantile=0.005,
#                 verbose=True,
#                 transforms=['Rigid', 'Affine', 'SyN'],
#                 transform_parameters=[(0.1,), (0.1,), (0.1, 3, 0)],
#                 metric=['MI', 'MI', 'CC'],
#                 metric_weight=[1, 1, 1],
#                 radius_or_number_of_bins=[32, 32, 32],
#                 sampling_strategy=['Regular', 'Regular', 'None'],
#                 sampling_percentage=[0.25, 0.25, None],
#                 number_of_iterations=[[1000, 500, 250, 100],
#                                       [1000, 500, 250, 100],
#                                       [100, 70, 50, 20]],
#                 convergence_threshold=[1e-6, 1e-6, 1e-6],
#                 convergence_window_size=[10, 10, 10],
#                 shrink_factors=[[8, 4, 2, 1],
#                                 [8, 4, 2, 1],
#                                 [8, 4, 2, 1]],
#                 smoothing_sigmas=[[3, 2, 1, 0],
#                                   [3, 2, 1, 0],
#                                   [3, 2, 1, 0]],
#                 output_warped_image=True),
#             inputs={
#                 'fixed_image': ('template_brain', nifti_gz_format),
#                 'moving_image': (self.brain_spec_name, nifti_gz_format)},
#             outputs={
#                 'mag_coreg_to_tmpl': ('warped_image', nifti_gz_format)},
#             wall_time=25,
#             requirements=[ants_req.v('2.0')])
#
#         select_trans = pipeline.add(
#             'select',
#             SelectOne(
#                 index=1),
#             inputs={
#                 'inlist': (ants_reg, 'forward_transforms')},
#             outputs={
#                 'coreg_to_tmpl_ants_mat': ('out', text_matrix_format)})
#
#         pipeline.add(
#             'select_warp',
#             SelectOne(
#                 index=0),
#             inputs={
#                 'inlist': (ants_reg, 'forward_transforms')},
#             outputs={
#                 'coreg_to_tmpl_ants_warp': ('out', nifti_gz_format)})
#
#         pipeline.add(
#             'slices',
#             FSLSlices(
#                 outname='coreg_to_tmpl_report'),
#             inputs={
#                 'im1': ('template', nifti_gz_format),
#                 'im2': (select_trans, 'out')},
#             outputs={
#                 'coreg_to_tmpl_fsl_report': ('report', gif_format)},
#             wall_time=1,
#             requirements=[fsl_req.v('5.0.8')])

        return pipeline

    def prepare_pipeline(self, **name_maps):
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
        pipeline = self.new_pipeline(
            name='prepare_pipeline',
            name_maps=name_maps,
            desc=("Dimensions swapping to ensure that all the images "
                  "have the same orientations."),
            citations=[fsl_cite])

        if (self.branch('reorient_to_std') or
                self.parameter('resampled_resolution') is not None):
            if self.branch('reorient_to_std'):
                swap = pipeline.add(
                    'fslreorient2std',
                    fsl.utils.Reorient2Std(
                        output_type='NIFTI_GZ'),
                    inputs={
                        'in_file': ('magnitude', nifti_gz_format)},
                    requirements=[fsl_req.v('5.0.9')])
    #         swap.inputs.new_dims = self.parameter('reoriented_dims')

            if self.parameter('resampled_resolution') is not None:
                resample = pipeline.add(
                    "resample",
                    MRResize(
                        voxel=self.parameter('resampled_resolution')),
                    inputs={
                        'in_file': (swap, 'out_file')},
                    requirements=[mrtrix_req.v('3.0rc3')])
                pipeline.connect_output('mag_preproc', resample, 'out_file',
                                        nifti_gz_format)
            else:
                pipeline.connect_output('mag_preproc', swap, 'out_file',
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
                    'mag_preproc': ('file', nifti_gz_format)})

        return pipeline

    def header_extraction_pipeline(self, **name_maps):

        pipeline = self.new_pipeline(
            name='header_extraction',
            name_maps=name_maps,
            desc=("Pipeline to extract the most important scan "
                  "information from the image header"),
            citations=[])

        input_format = self.input(self.header_image_spec_name).format

        if input_format == dicom_format:

            pipeline.add(
                'hd_info_extraction',
                DicomHeaderInfoExtraction(
                    multivol=False),
                inputs={
                    'dicom_folder': (self.header_image_spec_name, dicom_format)},
                outputs={
                    'tr': ('tr', float),
                    'start_time': ('start_time', str),
                    'total_duration': ('total_duration', str),
                    'real_duration': ('real_duration', str),
                    'ped': ('ped', str),
                    'pe_angle': ('pe_angle', str),
                    'echo_times': ('echo_times', float),
                    'voxel_sizes': ('voxel_sizes', float),
                    'main_field_strength': ('B0', float),
                    'main_field_orient': ('H', float)})

        elif input_format == nifti_gz_x_format:

            pipeline.add(
                'hd_info_extraction',
                NiftixHeaderInfoExtraction(),
                inputs={
                    'in_file': (self.header_image_spec_name, nifti_gz_x_format)},
                outputs={
                    'tr': ('tr', float),
                    'start_time': ('start_time', str),
                    'total_duration': ('total_duration', str),
                    'real_duration': ('real_duration', str),
                    'ped': ('ped', str),
                    'pe_angle': ('pe_angle', str),
                    'echo_times': ('echo_times', float),
                    'voxel_sizes': ('voxel_sizes', float),
                    'main_field_strength': ('B0', float),
                    'main_field_orient': ('H', float)})
        else:
            raise BananaUsageError(
                "Can only extract header info if 'magnitude' fileset "
                "is provided in DICOM or extended NIfTI format (provided {})"
                .format(self.input(self.header_image_spec_name).format))

        return pipeline

    def motion_mat_pipeline(self, **name_maps):

        pipeline = self.new_pipeline(
            name='motion_mat_calculation',
            name_maps=name_maps,
            desc=("Motion matrices calculation"),
            citations=[fsl_cite])

        mm = pipeline.add(
            'motion_mats',
            MotionMatCalculation(),
            outputs={
                'motion_mats': ('motion_mats', motion_mats_format)})
        if not self.spec('coreg_fsl_mat').derivable:
            logger.info("Cannot derive 'coreg_matrix' for {} required for "
                        "motion matrix calculation, assuming that it "
                        "is the reference study".format(self))
            mm.inputs.reference = True
            pipeline.connect_input('magnitude', mm, 'dummy_input')
        else:
            pipeline.connect_input('coreg_fsl_mat', mm, 'reg_mat',
                                   text_matrix_format)
            pipeline.connect_input('qform_mat', mm, 'qform_mat',
                                   text_matrix_format)
            if 'align_mats' in self.data_spec_names():
                pipeline.connect_input('align_mats', mm, 'align_mats',
                                       motion_mats_format)
        return pipeline
