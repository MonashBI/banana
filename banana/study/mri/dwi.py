from logging import getLogger
from nipype.interfaces.utility import Merge
from nipype.interfaces.mrtrix3 import ResponseSD, Tractography
from nipype.interfaces.mrtrix3.utils import BrainMask, TensorMetrics
from nipype.interfaces.mrtrix3.reconst import FitTensor, EstimateFOD
from banana.interfaces.mrtrix import (
    DWIPreproc, MRCat, ExtractDWIorB0, MRMath, DWIBiasCorrect, DWIDenoise,
    MRCalc, DWIIntensityNorm, AverageResponse, DWI2Mask)
# from nipype.workflows.dwi.fsl.tbss import create_tbss_all
# from banana.interfaces.noddi import (
#     CreateROI, BatchNODDIFitting, SaveParamsAsNIfTI)
from nipype.interfaces import fsl, mrtrix3, utility
from arcana.utils.interfaces import MergeTuple, Chain
from arcana.data import FilesetSpec, InputFilesetSpec
from arcana.utils.interfaces import SelectSession
from arcana.study import ParamSpec, SwitchSpec
from arcana.exceptions import ArcanaMissingDataException, ArcanaNameError
from banana.requirement import (
    fsl_req, mrtrix_req, ants_req)
from banana.interfaces.mrtrix import MRConvert, ExtractFSLGradients
from banana.study import StudyMetaClass
from banana.interfaces.custom.motion_correction import (
    PrepareDWI, AffineMatrixGeneration)
from banana.interfaces.custom.dwi import TransformGradients
from banana.interfaces.utility import AppendPath
from banana.study.base import Study
from banana.bids_ import BidsInputs, BidsAssocInput
from banana.exceptions import BananaUsageError
from banana.citation import (
    mrtrix_cite, fsl_cite, eddy_cite, topup_cite, distort_correct_cite,
    n4_cite, dwidenoise_cites)
from banana.file_format import (
    mrtrix_image_format, nifti_gz_format, nifti_gz_x_format, fsl_bvecs_format,
    fsl_bvals_format, text_format, dicom_format, eddy_par_format,
    mrtrix_track_format, motion_mats_format, text_matrix_format,
    directory_format, csv_format, zip_format)
from .base import MriStudy
from .epi import EpiSeriesStudy

logger = getLogger('banana')


class DwiStudy(EpiSeriesStudy, metaclass=StudyMetaClass):

    add_data_specs = [
        InputFilesetSpec('anat_5tt', mrtrix_image_format,
                         desc=("A co-registered segmentation image taken from "
                               "freesurfer output and simplified into 5 tissue"
                               " types. Used in ACT streamlines tractography"),
                         optional=True),
        InputFilesetSpec('anat_fs_recon_all', zip_format, optional=True,
                         desc=("Co-registered freesurfer recon-all output. "
                               "Used in building the connectome")),
        FilesetSpec('grad_dirs', fsl_bvecs_format, 'preprocess_pipeline'),
        FilesetSpec('grad_dirs_coreg', fsl_bvecs_format,
                    'series_coreg_pipeline',
                    desc=("The gradient directions coregistered to the "
                          "orientation of the coreg reference")),
        FilesetSpec('bvalues', fsl_bvals_format, 'preprocess_pipeline'),
        FilesetSpec('eddy_par', eddy_par_format, 'preprocess_pipeline'),
        FilesetSpec('noise_residual', mrtrix_image_format,
                    'preprocess_pipeline'),
        FilesetSpec('tensor', nifti_gz_format, 'tensor_pipeline'),
        FilesetSpec('fa', nifti_gz_format, 'tensor_metrics_pipeline'),
        FilesetSpec('adc', nifti_gz_format, 'tensor_metrics_pipeline'),
        FilesetSpec('wm_response', text_format, 'response_pipeline'),
        FilesetSpec('gm_response', text_format, 'response_pipeline'),
        FilesetSpec('csf_response', text_format, 'response_pipeline'),
        FilesetSpec('avg_response', text_format,
                    'average_response_pipeline'),
        FilesetSpec('wm_odf', mrtrix_image_format, 'fod_pipeline'),
        FilesetSpec('gm_odf', mrtrix_image_format, 'fod_pipeline'),
        FilesetSpec('csf_odf', mrtrix_image_format, 'fod_pipeline'),
        FilesetSpec('norm_intensity', mrtrix_image_format,
                    'intensity_normalisation_pipeline'),
        FilesetSpec('norm_intens_fa_template', mrtrix_image_format,
                    'intensity_normalisation_pipeline',
                    frequency='per_study'),
        FilesetSpec('norm_intens_wm_mask', mrtrix_image_format,
                    'intensity_normalisation_pipeline',
                    frequency='per_study'),
        FilesetSpec('global_tracks', mrtrix_track_format,
                    'global_tracking_pipeline'),
        FilesetSpec('wm_mask', mrtrix_image_format,
                    'global_tracking_pipeline'),
        FilesetSpec('connectome', csv_format, 'connectome_pipeline')]  # @IgnorePep8

    add_param_specs = [
        ParamSpec('multi_tissue', True),
        ParamSpec('preproc_pe_dir', None, dtype=str),
        ParamSpec('tbss_skel_thresh', 0.2),
        ParamSpec('fsl_mask_f', 0.25),
        ParamSpec('bet_robust', True),
        ParamSpec('bet_f_threshold', 0.2),
        ParamSpec('bet_reduce_bias', False),
        ParamSpec('num_global_tracks', int(1e9)),
        ParamSpec('global_tracks_cutoff', 0.05),
        SwitchSpec('preproc_denoise', False),
        SwitchSpec('response_algorithm', 'tax',
                   ('tax', 'dhollander', 'msmt_5tt')),
        SwitchSpec('fod_algorithm', 'csd', ('csd', 'msmt_csd')),
        MriStudy.param_spec('bet_method').with_new_choices('mrtrix'),
        SwitchSpec('reorient2std', False)]

    primary_bids_input = BidsInputs(
        spec_name='series', type='dwi',
        valid_formats=(nifti_gz_x_format, nifti_gz_format))

    default_bids_inputs = [primary_bids_input,
                           BidsAssocInput(
                               spec_name='bvalues',
                               primary=primary_bids_input,
                               association='grads',
                               type='bval',
                               format=fsl_bvals_format),
                           BidsAssocInput(
                               spec_name='grad_dirs',
                               primary=primary_bids_input,
                               association='grads',
                               type='bvec',
                               format=fsl_bvecs_format),
                           BidsAssocInput(
                               spec_name='reverse_phase',
                               primary=primary_bids_input,
                               association='epi',
                               format=nifti_gz_format,
                               drop_if_missing=True)]

    RECOMMENDED_NUM_SESSIONS_FOR_INTENS_NORM = 5

    @property
    def multi_tissue(self):
        return self.branch('response_algorithm',
                           ('msmt_5tt', 'dhollander'))

    def fsl_grads(self, pipeline, coregistered=True):
        "Adds and returns a node to the pipeline to merge the FSL grads and "
        "bvecs"
        try:
            fslgrad = pipeline.node('fslgrad')
        except ArcanaNameError:
            if self.is_coregistered and coregistered:
                grad_dirs = 'grad_dirs_coreg'
            else:
                grad_dirs = 'grad_dirs'
            # Gradient merge node
            fslgrad = pipeline.add(
                "fslgrad",
                MergeTuple(2),
                inputs={
                    'in1': (grad_dirs, fsl_bvecs_format),
                    'in2': ('bvalues', fsl_bvals_format)})
        return (fslgrad, 'out')

    def extract_magnitude_pipeline(self, **name_maps):

        pipeline = self.new_pipeline(
            'extract_magnitude',
            desc="Extracts the first b==0 volume from the series",
            citations=[],
            name_maps=name_maps)

        dwiextract = pipeline.add(
            'dwiextract',
            ExtractDWIorB0(
                bzero=True,
                out_ext='.nii.gz'),
            inputs={
                'in_file': ('series', nifti_gz_format),
                'fslgrad': self.fsl_grads(pipeline, coregistered=False)},
            requirements=[mrtrix_req.v('3.0rc3')])

        pipeline.add(
            "extract_first_vol",
            MRConvert(
                coord=(3, 0)),
            inputs={
                'in_file': (dwiextract, 'out_file')},
            outputs={
                'magnitude': ('out_file', nifti_gz_format)},
            requirements=[mrtrix_req.v('3.0rc3')])

        return pipeline

    def preprocess_pipeline(self, **name_maps):
        """
        Performs a series of FSL preprocessing steps, including Eddy and Topup

        Parameters
        ----------
        phase_dir : str{AP|LR|IS}
            The phase encode direction
        """

        # Determine whether we can correct for distortion, i.e. if reference
        # scans are provided
        # Include all references
        references = [fsl_cite, eddy_cite, topup_cite,
                      distort_correct_cite, n4_cite]
        if self.branch('preproc_denoise'):
            references.extend(dwidenoise_cites)

        pipeline = self.new_pipeline(
            name='preprocess',
            name_maps=name_maps,
            desc=(
                "Preprocess dMRI studies using distortion correction"),
            citations=references)

        # Create nodes to gradients to FSL format
        if self.input('series').format == dicom_format:
            extract_grad = pipeline.add(
                "extract_grad",
                ExtractFSLGradients(),
                inputs={
                    'in_file': ('series', dicom_format)},
                outputs={
                    'grad_dirs': ('bvecs_file', fsl_bvecs_format),
                    'bvalues': ('bvals_file', fsl_bvals_format)},
                requirements=[mrtrix_req.v('3.0rc3')])
            grad_fsl_inputs = {'in1': (extract_grad, 'bvecs_file'),
                               'in2': (extract_grad, 'bvals_file')}
        elif self.provided('grad_dirs') and self.provided('bvalues'):
            grad_fsl_inputs = {'in1': ('grad_dirs', fsl_bvecs_format),
                               'in2': ('bvalues', fsl_bvals_format)}
        else:
            raise BananaUsageError(
                "Either input 'magnitude' image needs to be in DICOM format "
                "or gradient directions and b-values need to be explicitly "
                "provided to {}".format(self))

        # Gradient merge node
        grad_fsl = pipeline.add(
            "grad_fsl",
            MergeTuple(2),
            inputs=grad_fsl_inputs)

        gradients = (grad_fsl, 'out')

        # Create node to reorient preproc out_file
        if self.branch('reorient2std'):
            reorient = pipeline.add(
                'fslreorient2std',
                fsl.utils.Reorient2Std(
                    output_type='NIFTI_GZ'),
                inputs={
                    'in_file': ('series', nifti_gz_format)},
                requirements=[fsl_req.v('5.0.9')])
            reoriented = (reorient, 'out_file')
        else:
            reoriented = ('series', nifti_gz_format)

        # Denoise the dwi-scan
        if self.branch('preproc_denoise'):
            # Run denoising
            denoise = pipeline.add(
                'denoise',
                DWIDenoise(),
                inputs={
                    'in_file': reoriented},
                requirements=[mrtrix_req.v('3.0rc3')])

            # Calculate residual noise
            subtract_operands = pipeline.add(
                'subtract_operands',
                Merge(2),
                inputs={
                    'in1': reoriented,
                    'in2': (denoise, 'noise')})

            pipeline.add(
                'subtract',
                MRCalc(
                    operation='subtract'),
                inputs={
                    'operands': (subtract_operands, 'out')},
                outputs={
                    'noise_residual': ('out_file', mrtrix_image_format)},
                requirements=[mrtrix_req.v('3.0rc3')])
            denoised = (denoise, 'out_file')
        else:
            denoised = reoriented

        # Preproc kwargs
        preproc_kwargs = {}
        preproc_inputs = {'in_file': denoised,
                          'grad_fsl': gradients}

        if self.provided('reverse_phase'):

            if self.provided('magnitude', default_okay=False):
                dwi_reference = ('magnitude', mrtrix_image_format)
            else:
                # Extract b=0 volumes
                dwiextract = pipeline.add(
                    'dwiextract',
                    ExtractDWIorB0(
                        bzero=True,
                        out_ext='.nii.gz'),
                    inputs={
                        'in_file': denoised,
                        'fslgrad': gradients},
                    requirements=[mrtrix_req.v('3.0rc3')])

                # Get first b=0 from dwi b=0 volumes
                extract_first_b0 = pipeline.add(
                    "extract_first_vol",
                    MRConvert(
                        coord=(3, 0)),
                    inputs={
                        'in_file': (dwiextract, 'out_file')},
                    requirements=[mrtrix_req.v('3.0rc3')])

                dwi_reference = (extract_first_b0, 'out_file')

            # Concatenate extracted forward rpe with reverse rpe
            combined_images = pipeline.add(
                'combined_images',
                MRCat(),
                inputs={
                    'first_scan': dwi_reference,
                    'second_scan': ('reverse_phase', mrtrix_image_format)},
                requirements=[mrtrix_req.v('3.0rc3')])

            # Create node to assign the right PED to the diffusion
            prep_dwi = pipeline.add(
                'prepare_dwi',
                PrepareDWI(),
                inputs={
                    'pe_dir': ('ped', float),
                    'ped_polarity': ('pe_angle', float)})

            preproc_kwargs['rpe_pair'] = True

            distortion_correction = True
            preproc_inputs['se_epi'] = (combined_images, 'out_file')
        else:
            distortion_correction = False
            preproc_kwargs['rpe_none'] = True

        if self.parameter('preproc_pe_dir') is not None:
            preproc_kwargs['pe_dir'] = self.parameter('preproc_pe_dir')

        preproc = pipeline.add(
            'dwipreproc',
            DWIPreproc(
                no_clean_up=True,
                out_file_ext='.nii.gz',
                # FIXME: Need to determine this programmatically
                # eddy_parameters = '--data_is_shelled '
                temp_dir='dwipreproc_tempdir',
                **preproc_kwargs),
            inputs=preproc_inputs,
            outputs={
                'eddy_par': ('eddy_parameters', eddy_par_format)},
            requirements=[mrtrix_req.v('3.0rc3'), fsl_req.v('5.0.10')],
            wall_time=60)

        if distortion_correction:
            pipeline.connect(prep_dwi, 'pe', preproc, 'pe_dir')

        mask = pipeline.add(
            'dwi2mask',
            BrainMask(
                out_file='brainmask.nii.gz'),
            inputs={
                'in_file': (preproc, 'out_file'),
                'grad_fsl': gradients},
            requirements=[mrtrix_req.v('3.0rc3')])

        # Create bias correct node
        pipeline.add(
            "bias_correct",
            DWIBiasCorrect(
                method='ants'),
            inputs={
                'grad_fsl': gradients,  # internal
                'in_file': (preproc, 'out_file'),
                'mask': (mask, 'out_file')},
            outputs={
                'series_preproc': ('out_file', nifti_gz_format)},
            requirements=[mrtrix_req.v('3.0rc3'), ants_req.v('2.0')])

        return pipeline

    def brain_extraction_pipeline(self, **name_maps):
        """
        Generates a whole brain mask using MRtrix's 'dwi2mask' command

        Parameters
        ----------
        mask_tool: Str
            Can be either 'bet' or 'dwi2mask' depending on which mask tool you
            want to use
        """

        if self.branch('bet_method', 'mrtrix'):
            pipeline = self.new_pipeline(
                'brain_extraction',
                desc="Generate brain mask from b0 images",
                citations=[mrtrix_cite],
                name_maps=name_maps)

            if self.provided('coreg_ref'):
                series = 'series_coreg'
            else:
                series = 'series_preproc'

            # Create mask node
            masker = pipeline.add(
                'dwi2mask',
                BrainMask(
                    out_file='brain_mask.nii.gz'),
                inputs={
                    'in_file': (series, nifti_gz_format),
                    'grad_fsl': self.fsl_grads(pipeline, coregistered=False)},
                outputs={
                    'brain_mask': ('out_file', nifti_gz_format)},
                requirements=[mrtrix_req.v('3.0rc3')])

            merge = pipeline.add(
                'merge_operands',
                Merge(2),
                inputs={
                    'in1': ('mag_preproc', nifti_gz_format),
                    'in2': (masker, 'out_file')})

            pipeline.add(
                'apply_mask',
                MRCalc(
                    operation='multiply'),
                inputs={
                    'operands': (merge, 'out')},
                outputs={
                    'brain': ('out_file', nifti_gz_format)},
                requirements=[mrtrix_req.v('3.0rc3')])
        else:
            pipeline = super().brain_extraction_pipeline(**name_maps)
        return pipeline

    def series_coreg_pipeline(self, **name_maps):

        pipeline = super().series_coreg_pipeline(**name_maps)

        # Apply coregistration transform to gradients
        pipeline.add(
            'transform_grads',
            TransformGradients(),
            inputs={
                'gradients': ('grad_dirs', fsl_bvecs_format),
                'transform': ('coreg_fsl_mat', text_matrix_format)},
            outputs={
                'grad_dirs_coreg': ('transformed', fsl_bvecs_format)})

        return pipeline

    def intensity_normalisation_pipeline(self, **name_maps):

        if self.num_sessions < 2:
            raise ArcanaMissingDataException(
                "Cannot normalise intensities of DWI images as study only "
                "contains a single session")
        elif self.num_sessions < self.RECOMMENDED_NUM_SESSIONS_FOR_INTENS_NORM:
            logger.warning(
                "The number of sessions in the study ({}) is less than the "
                "recommended number for intensity normalisation ({}). The "
                "results may be unreliable".format(
                    self.num_sessions,
                    self.RECOMMENDED_NUM_SESSIONS_FOR_INTENS_NORM))

        pipeline = self.new_pipeline(
            name='intensity_normalization',
            desc="Corrects for B1 field inhomogeneity",
            citations=[mrtrix_req.v('3.0rc3')],
            name_maps=name_maps)

        mrconvert = pipeline.add(
            'mrconvert',
            MRConvert(
                out_ext='.mif'),
            inputs={
                'in_file': (self.series_preproc_spec_name, nifti_gz_format),
                'grad_fsl': self.fsl_grads(pipeline)},
            requirements=[mrtrix_req.v('3.0rc3')])

        # Pair subject and visit ids together, expanding so they can be
        # joined and chained together
        session_ids = pipeline.add(
            'session_ids',
            utility.IdentityInterface(
                ['subject_id', 'visit_id']),
            inputs={
                'subject_id': (Study.SUBJECT_ID, int),
                'visit_id': (Study.VISIT_ID, int)})

        # Set up join nodes
        join_fields = ['dwis', 'masks', 'subject_ids', 'visit_ids']
        join_over_subjects = pipeline.add(
            'join_over_subjects',
            utility.IdentityInterface(
                join_fields),
            inputs={
                'masks': (self.brain_mask_spec_name, nifti_gz_format),
                'dwis': (mrconvert, 'out_file'),
                'subject_ids': (session_ids, 'subject_id'),
                'visit_ids': (session_ids, 'visit_id')},
            joinsource=self.SUBJECT_ID,
            joinfield=join_fields)

        join_over_visits = pipeline.add(
            'join_over_visits',
            Chain(
                join_fields),
            inputs={
                'dwis': (join_over_subjects, 'dwis'),
                'masks': (join_over_subjects, 'masks'),
                'subject_ids': (join_over_subjects, 'subject_ids'),
                'visit_ids': (join_over_subjects, 'visit_ids')},
            joinsource=self.VISIT_ID,
            joinfield=join_fields)

        # Intensity normalization
        intensity_norm = pipeline.add(
            'dwiintensitynorm',
            DWIIntensityNorm(),
            inputs={
                'in_files': (join_over_visits, 'dwis'),
                'masks': (join_over_visits, 'masks')},
            outputs={
                'norm_intens_fa_template': ('fa_template',
                                            mrtrix_image_format),
                'norm_intens_wm_mask': ('wm_mask', mrtrix_image_format)},
            requirements=[mrtrix_req.v('3.0rc3')])

        # Set up expand nodes
        pipeline.add(
            'expand', SelectSession(),
            inputs={
                'subject_ids': (join_over_visits, 'subject_ids'),
                'visit_ids': (join_over_visits, 'visit_ids'),
                'inlist': (intensity_norm, 'out_files'),
                'subject_id': (Study.SUBJECT_ID, int),
                'visit_id': (Study.VISIT_ID, int)},
            outputs={
                'norm_intensity': ('item', mrtrix_image_format)})

        # Connect inputs
        return pipeline

    def tensor_pipeline(self, **name_maps):  # @UnusedVariable
        """
        Fits the apparrent diffusion tensor (DT) to each voxel of the image
        """

        pipeline = self.new_pipeline(
            name='tensor',
            desc=("Estimates the apparent diffusion tensor in each "
                  "voxel"),
            citations=[],
            name_maps=name_maps)

        # Create tensor fit node
        pipeline.add(
            'dwi2tensor',
            FitTensor(
                out_file='dti.nii.gz'),
            inputs={
                'grad_fsl': self.fsl_grads(pipeline),
                'in_file': (self.series_preproc_spec_name, nifti_gz_format),
                'in_mask': (self.brain_mask_spec_name, nifti_gz_format)},
            outputs={
                'tensor': ('out_file', nifti_gz_format)},
            requirements=[mrtrix_req.v('3.0rc3')])

        return pipeline

    def tensor_metrics_pipeline(self, **name_maps):  # @UnusedVariable
        """
        Fits the apparrent diffusion tensor (DT) to each voxel of the image
        """

        pipeline = self.new_pipeline(
            name='fa',
            desc=("Calculates the FA and ADC from a tensor image"),
            citations=[],
            name_maps=name_maps)

        # Create tensor fit node
        pipeline.add(
            'metrics',
            TensorMetrics(
                out_fa='fa.nii.gz',
                out_adc='adc.nii.gz'),
            inputs={
                'in_file': ('tensor', nifti_gz_format),
                'in_mask': (self.brain_mask_spec_name, nifti_gz_format)},
            outputs={
                'fa': ('out_fa', nifti_gz_format),
                'adc': ('out_adc', nifti_gz_format)},
            requirements=[mrtrix_req.v('3.0rc3')])

        return pipeline

    def response_pipeline(self, **name_maps):  # @UnusedVariable
        """
        Estimates the fibre orientation distribution (FOD) using constrained
        spherical deconvolution

        Parameters
        ----------
        response_algorithm : str
            Algorithm used to estimate the response
        """

        pipeline = self.new_pipeline(
            name='response',
            desc=("Estimates the fibre response function"),
            citations=[mrtrix_cite],
            name_maps=name_maps)

        # Create fod fit node
        response = pipeline.add(
            'response',
            ResponseSD(
                algorithm=self.parameter('response_algorithm')),
            inputs={
                'grad_fsl': self.fsl_grads(pipeline),
                'in_file': (self.series_preproc_spec_name, nifti_gz_format),
                'in_mask': (self.brain_mask_spec_name, nifti_gz_format)},
            outputs={
                'wm_response': ('wm_file', text_format)},
            requirements=[mrtrix_req.v('3.0rc3')])

        # Connect to outputs
        if self.multi_tissue:
            response.inputs.gm_file = 'gm.txt',
            response.inputs.csf_file = 'csf.txt',
            pipeline.connect_output('gm_response', response, 'gm_file',
                                    text_format)
            pipeline.connect_output('csf_response', response, 'csf_file',
                                    text_format)

        return pipeline

    def average_response_pipeline(self, **name_maps):
        """
        Averages the estimate response function over all subjects in the
        project
        """

        pipeline = self.new_pipeline(
            name='average_response',
            desc=(
                "Averages the fibre response function over the project"),
            citations=[mrtrix_cite],
            name_maps=name_maps)

        join_subjects = pipeline.add(
            'join_subjects',
            utility.IdentityInterface(['responses']),
            inputs={
                'responses': ('wm_response', text_format)},
            outputs={},
            joinsource=self.SUBJECT_ID,
            joinfield=['responses'])

        join_visits = pipeline.add(
            'join_visits',
            Chain(['responses']),
            inputs={
                'responses': (join_subjects, 'responses')},
            joinsource=self.VISIT_ID,
            joinfield=['responses'])

        pipeline.add(
            'avg_response',
            AverageResponse(),
            inputs={
                'in_files': (join_visits, 'responses')},
            outputs={
                'avg_response': ('out_file', text_format)},
            requirements=[mrtrix_req.v('3.0rc3')])

        return pipeline

    def fod_pipeline(self, **name_maps):  # @UnusedVariable
        """
        Estimates the fibre orientation distribution (FOD) using constrained
        spherical deconvolution

        Parameters
        ----------
        """

        pipeline = self.new_pipeline(
            name='fod',
            desc=("Estimates the fibre orientation distribution in each"
                  " voxel"),
            citations=[mrtrix_cite],
            name_maps=name_maps)

        if self.branch('fod_algorithm', 'msmt_csd'):
            pipeline.add_input(FilesetSpec('gm_response', text_format))
            pipeline.add_input(FilesetSpec('csf_response', text_format))

        # Create fod fit node
        dwi2fod = pipeline.add(
            'dwi2fod',
            EstimateFOD(
                algorithm=self.parameter('fod_algorithm')),
            inputs={
                'in_file': (self.series_preproc_spec_name, nifti_gz_format),
                'wm_txt': ('wm_response', text_format),
                'mask_file': (self.brain_mask_spec_name, nifti_gz_format),
                'grad_fsl': self.fsl_grads(pipeline)},
            outputs={
                'wm_odf': ('wm_odf', nifti_gz_format)},
            requirements=[mrtrix_req.v('3.0rc3')])

        if self.multi_tissue:
            dwi2fod.inputs.gm_odf = 'gm.mif',
            dwi2fod.inputs.csf_odf = 'csf.mif',
            pipeline.connect_input('gm_response', dwi2fod, 'gm_txt',
                                   text_format),
            pipeline.connect_input('csf_response', dwi2fod, 'csf_txt',
                                   text_format),
            pipeline.connect_output('gm_odf', dwi2fod, 'gm_odf',
                                    nifti_gz_format),
            pipeline.connect_output('csf_odf', dwi2fod, 'csf_odf',
                                    nifti_gz_format),
        # Check inputs/output are connected
        return pipeline

    def extract_b0_pipeline(self, **name_maps):  # @UnusedVariable
        """
        Extracts the b0 images from a DWI study and takes their mean
        """

        pipeline = self.new_pipeline(
            name='extract_b0',
            desc="Extract b0 image from a DWI study",
            citations=[mrtrix_cite],
            name_maps=name_maps)

        # Extraction node
        extract_b0s = pipeline.add(
            'extract_b0s',
            ExtractDWIorB0(
                bzero=True,
                quiet=True),
            inputs={
                'fslgrad': self.fsl_grads(pipeline),
                'in_file': (self.series_preproc_spec_name, nifti_gz_format)},
            requirements=[mrtrix_req.v('3.0rc3')])

        # FIXME: Need a registration step before the mean
        # Mean calculation node
        mean = pipeline.add(
            "mean",
            MRMath(
                axis=3,
                operation='mean',
                quiet=True),
            inputs={
                'in_files': (extract_b0s, 'out_file')},
            requirements=[mrtrix_req.v('3.0rc3')])

        # Convert to Nifti
        pipeline.add(
            "output_conversion",
            MRConvert(
                out_ext='.nii.gz',
                quiet=True),
            inputs={
                'in_file': (mean, 'out_file')},
            outputs={
                'b0': ('out_file', nifti_gz_format)},
            requirements=[mrtrix_req.v('3.0rc3')])

        return pipeline

    def global_tracking_pipeline(self, **name_maps):

        pipeline = self.new_pipeline(
            name='global_tracking',
            desc="Extract b0 image from a DWI study",
            citations=[mrtrix_cite],
            name_maps=name_maps)

        mask = pipeline.add(
            'mask',
            DWI2Mask(),
            inputs={
                'grad_fsl': self.fsl_grads(pipeline),
                'in_file': (self.series_preproc_spec_name, nifti_gz_format)},
            requirements=[mrtrix_req.v('3.0rc3')])

        tracking = pipeline.add(
            'tracking',
            Tractography(
                select=self.parameter('num_global_tracks'),
                cutoff=self.parameter('global_tracks_cutoff')),
            inputs={
                'seed_image': (mask, 'out_file'),
                'in_file': ('wm_odf', mrtrix_image_format)},
            outputs={
                'global_tracks': ('out_file', mrtrix_track_format)},
            requirements=[mrtrix_req.v('3.0rc3')])

        if self.provided('anat_5tt'):
            pipeline.connect_input('anat_5tt', tracking, 'act_file',
                                   mrtrix_image_format)

        return pipeline

    def intrascan_alignment_pipeline(self, **name_maps):

        pipeline = self.new_pipeline(
            name='affine_mat_generation',
            desc=("Generation of the affine matrices for the main dwi "
                  "sequence starting from eddy motion parameters"),
            citations=[fsl_cite],
            name_maps=name_maps)

        pipeline.add(
            'gen_aff_mats',
            AffineMatrixGeneration(),
            inputs={
                'reference_image': ('mag_preproc', nifti_gz_format),
                'motion_parameters': ('eddy_par', eddy_par_format)},
            outputs={
                'align_mats': ('affine_matrices', motion_mats_format)})

        return pipeline

    def connectome_pipeline(self, **name_maps):

        pipeline = self.new_pipeline(
            name='connectome',
            desc=("Generate a connectome from whole brain connectivity"),
            citations=[],
            name_maps=name_maps)

        aseg_path = pipeline.add(
            'aseg_path',
            AppendPath(
                sub_paths=['mri', 'aparc+aseg.mgz']),
            inputs={
                'base_path': ('anat_fs_recon_all', directory_format)})

        pipeline.add(
            'connectome',
            mrtrix3.BuildConnectome(),
            inputs={
                'in_file': ('global_tracks', mrtrix_track_format),
                'in_parc': (aseg_path, 'out_path')},
            outputs={
                'connectome': ('out_file', csv_format)},
            requirements=[mrtrix_req.v('3.0rc3')])

        return pipeline
