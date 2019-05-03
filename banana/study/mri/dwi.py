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
from banana.interfaces.mrtrix import MRConvert, ExtractFSLGradients
from arcana.utils.interfaces import MergeTuple, Chain
from nipype.interfaces.utility import IdentityInterface
from banana.citation import (
    mrtrix_cite, fsl_cite, eddy_cite, topup_cite, distort_correct_cite,
    fast_cite, n4_cite, dwidenoise_cites)
from banana.file_format import (
    mrtrix_image_format, nifti_gz_format, nifti_gz_x_format, fsl_bvecs_format,
    fsl_bvals_format, text_format, dicom_format, eddy_par_format,
    mrtrix_track_format, motion_mats_format)
from banana.requirement import (
    fsl_req, mrtrix_req, ants_req)
from arcana.data import FilesetSpec, InputFilesetSpec
from arcana.utils.interfaces import SelectSession
from arcana.study import ParamSpec, SwitchSpec
from .epi import EpiStudy
from nipype.interfaces import fsl
from banana.interfaces.custom.motion_correction import (
    PrepareDWI, AffineMatrixGeneration)
from banana.study.base import Study
from banana.bids import BidsInput, BidsAssocInput
from banana.exceptions import BananaUsageError
from arcana.exceptions import ArcanaMissingDataException
from banana.study import StudyMetaClass

from banana.file_format import STD_IMAGE_FORMATS

logger = getLogger('banana')


class DwiStudy(EpiStudy, metaclass=StudyMetaClass):

    add_data_specs = [
        InputFilesetSpec('dwi_reference', STD_IMAGE_FORMATS, optional=True),
        FilesetSpec('b0', nifti_gz_format, 'extract_b0_pipeline',
                    desc="b0 image"),
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
        FilesetSpec('bias_correct', nifti_gz_format,
                    'bias_correct_pipeline'),
        FilesetSpec('grad_dirs', fsl_bvecs_format, 'preprocess_pipeline'),
        FilesetSpec('bvalues', fsl_bvals_format, 'preprocess_pipeline'),
        FilesetSpec('eddy_par', eddy_par_format, 'preprocess_pipeline'),
        FilesetSpec('brain', nifti_gz_format,
                    'brain_extraction_pipeline'),
        FilesetSpec('brain_mask', nifti_gz_format,
                    'brain_extraction_pipeline'),
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
                    'global_tracking_pipeline')
        # FilesetSpec('tbss_mean_fa', nifti_gz_format, 'tbss_pipeline',
        #             frequency='per_study'),
        # FilesetSpec('tbss_proj_fa', nifti_gz_format, 'tbss_pipeline',
        #             frequency='per_study'),
        # FilesetSpec('tbss_skeleton', nifti_gz_format, 'tbss_pipeline',
        #             frequency='per_study'),
        # FilesetSpec('tbss_skeleton_mask', nifti_gz_format,
        #             'tbss_pipeline', frequency='per_study'),
        ]  # @IgnorePep8

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
        SwitchSpec('brain_extract_method', 'mrtrix',
                   ('mrtrix', 'fsl')),
        SwitchSpec('bias_correct_method', 'ants', ('ants', 'fsl'))]

    primary_bids_input = BidsInput(
        spec_name='magnitude', type='dwi', format=nifti_gz_x_format)

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
                      distort_correct_cite]
        if self.branch('preproc_denoise'):
            references.extend(dwidenoise_cites)

        pipeline = self.new_pipeline(
            name='preprocess',
            name_maps=name_maps,
            desc=(
                "Preprocess dMRI studies using distortion correction"),
            citations=references)

        # Create nodes to gradients to FSL format
        if self.input('magnitude').format == dicom_format:
            extract_grad = pipeline.add(
                "extract_grad",
                ExtractFSLGradients(),
                inputs={
                    'in_file': ('magnitude', dicom_format)},
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

        # Denoise the dwi-scan
        if self.branch('preproc_denoise'):
            # Run denoising
            denoise = pipeline.add(
                'denoise',
                DWIDenoise(),
                inputs={
                    'in_file': ('magnitude', nifti_gz_format)},
                requirements=[mrtrix_req.v('3.0rc3')])

            # Calculate residual noise
            subtract_operands = pipeline.add(
                'subtract_operands',
                Merge(2),
                inputs={
                    'in1': ('magnitude', nifti_gz_format),
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

        # Preproc kwargs
        preproc_kwargs = {}
        preproc_inputs = {'grad_fsl': (grad_fsl, 'out')}

        if (self.provided('dwi_reference') or
                self.provided('reverse_phase')):
            # Extract b=0 volumes
            dwiextract = pipeline.add(
                'dwiextract',
                ExtractDWIorB0(
                    bzero=True,
                    out_ext='.nii.gz'),
                inputs={
                    'in_file': ('magnitude', dicom_format)},
                requirements=[mrtrix_req.v('3.0rc3')])

            # Get first b=0 from dwi b=0 volumes
            mrconvert = pipeline.add(
                "mrconvert",
                MRConvert(
                    coord=(3, 0)),
                inputs={
                    'in_file': (dwiextract, 'out_file')},
                requirements=[mrtrix_req.v('3.0rc3')])

            # Concatenate extracted forward rpe with reverse rpe
            mrcat = pipeline.add(
                'mrcat',
                MRCat(),
                inputs={
                    'second_scan': ((
                        'dwi_reference' if self.provided('dwi_reference')
                        else 'reverse_phase'), mrtrix_image_format),
                    'first_scan': (mrconvert, 'out_file')},
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
            preproc_inputs['se_epi'] = (mrcat, 'out_file')
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
        if self.branch('preproc_denoise'):
            pipeline.connect(denoise, 'out_file', preproc, 'in_file')
        else:
            pipeline.connect_input('magnitude', preproc, 'in_file',
                                   nifti_gz_format)
        if distortion_correction:
            pipeline.connect(prep_dwi, 'pe', preproc, 'pe_dir')

        # Create node to reorient preproc out_file
        pipeline.add(
            'fslreorient2std',
            fsl.utils.Reorient2Std(),
            inputs={
                'in_file': (preproc, 'out_file')},
            outputs={
                'preproc': ('out_file', nifti_gz_format)},
            requirements=[fsl_req.v('5.0.9')])

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

        if self.branch('brain_extract_method', 'mrtrix'):
            pipeline = self.new_pipeline(
                'brain_extraction',
                desc="Generate brain mask from b0 images",
                citations=[mrtrix_cite],
                name_maps=name_maps)

            # Gradient merge node
            grad_fsl = pipeline.add(
                "grad_fsl",
                MergeTuple(2),
                inputs={
                    'in1': ('grad_dirs', fsl_bvecs_format),
                    'in2': ('bvalues', fsl_bvals_format)})

            # Create mask node
            pipeline.add(
                'dwi2mask',
                BrainMask(
                    out_file='brain_mask.nii.gz'),
                inputs={
                    'in_file': ('preproc', nifti_gz_format),
                    'grad_fsl': (grad_fsl, 'out')},
                outputs={
                    'brain_mask': ('out_file', nifti_gz_format)},
                requirements=[mrtrix_req.v('3.0rc3')])

        else:
            pipeline = super(DwiStudy, self).brain_extraction_pipeline(
                **name_maps)
        return pipeline

    def bias_correct_pipeline(self, **name_maps):
        """
        Corrects B1 field inhomogeneities
        """

        pipeline = self.new_pipeline(
            name='bias_correct',
            desc="Corrects for B1 field inhomogeneity",
            citations=[fast_cite,
                        (n4_cite
                         if self.parameter('bias_correct_method') == 'ants'
                         else fsl_cite)],
            name_maps=name_maps)

        # Gradient merge node
        fsl_grads = pipeline.add(
            "fsl_grads",
            MergeTuple(2),
            inputs={
                'in1': ('grad_dirs', fsl_bvecs_format),
                'in2': ('bvalues', fsl_bvals_format)})

        # Create bias correct node
        pipeline.add(
            "bias_correct",
            DWIBiasCorrect(
                method=self.parameter('bias_correct_method')),
            inputs={
                'grad_fsl': (fsl_grads, 'out'),  # internal
                'in_file': ('preproc', nifti_gz_format),
                'mask': ('brain_mask', nifti_gz_format)},
            outputs={
                'bias_correct': ('out_file', nifti_gz_format)},
            requirements=(
                [mrtrix_req.v('3.0rc3'),
                 (ants_req.v('2.0')
                  if self.parameter('bias_correct_method') == 'ants'
                  else fsl_req.v('5.0.9'))]))

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

        # Convert from nifti to mrtrix format
        grad_merge = pipeline.add(
            "grad_merge",
            MergeTuple(2),
            inputs={
                'in1': ('grad_dirs', fsl_bvecs_format),
                'in2': ('bvalues', fsl_bvals_format)})

        mrconvert = pipeline.add(
            'mrconvert',
            MRConvert(
                out_ext='.mif'),
            inputs={
                'in_file': ('bias_correct', nifti_gz_format),
                'grad_fsl': (grad_merge, 'out')})

        # Pair subject and visit ids together, expanding so they can be
        # joined and chained together
        session_ids = pipeline.add(
            'session_ids',
            IdentityInterface(
                ['subject_id', 'visit_id']),
            inputs={
                'subject_id': (Study.SUBJECT_ID, int),
                'visit_id': (Study.VISIT_ID, int)})

        # Set up join nodes
        join_fields = ['dwis', 'masks', 'subject_ids', 'visit_ids']
        join_over_subjects = pipeline.add(
            'join_over_subjects',
            IdentityInterface(
                join_fields),
            inputs={
                'masks': ('brain_mask', nifti_gz_format),
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
                'norm_intens_fa_template': ('fa_template', mrtrix_image_format),
                'norm_intens_wm_mask': ('wm_mask', mrtrix_image_format)})

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

        # Gradient merge node
        fsl_grads = pipeline.add(
            "fsl_grads",
            MergeTuple(2),
            inputs={
                'in1': ('grad_dirs', fsl_bvecs_format),
                'in2': ('bvalues', fsl_bvals_format)})

        # Create tensor fit node
        pipeline.add(
            'dwi2tensor',
            FitTensor(
                out_file='dti.nii.gz'),
            inputs={
                'grad_fsl': (fsl_grads, 'out'),
                'in_file': ('bias_correct', nifti_gz_format),
                'in_mask': ('brain_mask', nifti_gz_format)},
            outputs={
                'tensor': ('out_file', nifti_gz_format)})

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
                'in_mask': ('brain_mask', nifti_gz_format)},
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

        # Gradient merge node
        fsl_grads = pipeline.add(
            "fsl_grads",
            MergeTuple(2),
            inputs={
                'in1': ('grad_dirs', fsl_bvecs_format),
                'in2': ('bvalues', fsl_bvals_format)})

        # Create fod fit node
        response = pipeline.add(
            'response',
            ResponseSD(
                algorithm=self.parameter('response_algorithm')),
            inputs={
                'grad_fsl': (fsl_grads, 'out'),
                'in_file': ('bias_correct', nifti_gz_format),
                'in_mask': ('brain_mask', nifti_gz_format)},
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
            IdentityInterface(['responses']),
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
                'avg_response': ('out_file', text_format)})

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

        # Gradient merge node
        fsl_grads = pipeline.add(
            "fsl_grads",
            MergeTuple(2),
            inputs={
                'in1': ('grad_dirs', fsl_bvecs_format),
                'in2': ('bvalues', fsl_bvals_format)})

        # Create fod fit node
        dwi2fod = pipeline.add(
            'dwi2fod',
            EstimateFOD(
                algorithm=self.parameter('fod_algorithm')),
            inputs={
                'in_file': ('bias_correct', nifti_gz_format),
                'wm_txt': ('wm_response', text_format),
                'mask_file': ('brain_mask', nifti_gz_format),
                'grad_fsl': (fsl_grads, 'out')},
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

        # Gradient merge node
        fsl_grads = pipeline.add(
            "fsl_grads",
            MergeTuple(2),
            inputs={
                'in1': ('grad_dirs', fsl_bvecs_format),
                'in2': ('bvalues', fsl_bvals_format)})

        # Extraction node
        extract_b0s = pipeline.add(
            'extract_b0s',
            ExtractDWIorB0(
                bzero=True,
                quiet=True),
            inputs={
                'fslgrad': (fsl_grads, 'out'),
                'in_file': ('bias_correct', nifti_gz_format)},
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

        # Add gradients to input image
        fsl_grads = pipeline.add(
            "fsl_grads",
            MergeTuple(2),
            inputs={
                'in1': ('grad_dirs', fsl_bvecs_format),
                'in2': ('bvalues', fsl_bvals_format)})

        mask = pipeline.add(
            'mask',
            DWI2Mask(),
            inputs={
                'grad_fsl': (fsl_grads, 'out'),
                'in_file': ('bias_correct', nifti_gz_format)})

        pipeline.add(
            'tracking',
            Tractography(
                select=self.parameter('num_global_tracks'),
                cutoff=self.parameter('global_tracks_cutoff')),
            inputs={
                'seed_image': (mask, 'out_file'),
                'in_file': ('wm_odf', mrtrix_image_format)},
            outputs={
                'global_tracks': ('out_file', mrtrix_track_format)})

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
                'reference_image': ('preproc', nifti_gz_format),
                'motion_parameters': ('eddy_par', eddy_par_format)},
            outputs={
                'align_mats': ('affine_matrices', motion_mats_format)})

        return pipeline

#     def tbss_pipeline(self, **name_maps):  # @UnusedVariable
#
# #             inputs=[FilesetSpec('fa', nifti_gz_format)],
# #             outputs=[FilesetSpec('tbss_mean_fa', nifti_gz_format),
# #                      FilesetSpec('tbss_proj_fa', nifti_gz_format,
# #                                  frequency='per_study'),
# #                      FilesetSpec('tbss_skeleton', nifti_gz_format,
# #                                  frequency='per_study'),
# #                      FilesetSpec('tbss_skeleton_mask', nifti_gz_format,
# #                                  frequency='per_study')],
#         pipeline = self.new_pipeline(
#             name='tbss',
#             citations=[tbss_cite, fsl_cite],
#             name_maps=name_maps)
#         # Create TBSS workflow
# #         tbss = create_tbss_all(name='tbss')
#         # Connect inputs
#         pipeline.connect_input('fa', tbss, 'inputnode.fa_list')
#         # Connect outputs
#         pipeline.connect_output('tbss_mean_fa', tbss,
#                                 'outputnode.meanfa_file')
#         pipeline.connect_output('tbss_proj_fa', tbss,
#                                 'outputnode.projectedfa_file')
#         pipeline.connect_output('tbss_skeleton', tbss,
#                                 'outputnode.skeleton_file')
#         pipeline.connect_output('tbss_skeleton_mask', tbss,
#                                 'outputnode.skeleton_mask')
#         # Check inputs/output are connected
#         return pipeline


# class NODDIStudy(DwiStudy, metaclass=StudyMetaClass):
# 
#     add_data_specs = [
#         InputFilesetSpec('low_b_dw_scan', mrtrix_image_format),
#         InputFilesetSpec('high_b_dw_scan', mrtrix_image_format),
#         FilesetSpec('dwi_scan', mrtrix_image_format, 'concatenate_pipeline'),
#         FilesetSpec('ficvf', nifti_format, 'noddi_fitting_pipeline'),
#         FilesetSpec('odi', nifti_format, 'noddi_fitting_pipeline'),
#         FilesetSpec('fiso', nifti_format, 'noddi_fitting_pipeline'),
#         FilesetSpec('fibredirs_xvec', nifti_format, 'noddi_fitting_pipeline'),
#         FilesetSpec('fibredirs_yvec', nifti_format, 'noddi_fitting_pipeline'),
#         FilesetSpec('fibredirs_zvec', nifti_format, 'noddi_fitting_pipeline'),
#         FilesetSpec('fmin', nifti_format, 'noddi_fitting_pipeline'),
#         FilesetSpec('kappa', nifti_format, 'noddi_fitting_pipeline'),
#         FilesetSpec('error_code', nifti_format, 'noddi_fitting_pipeline')]
# 
#     add_param_specs = [ParamSpec('noddi_model',
#                                          'WatsonSHStickTortIsoV_B0'),
#                            SwitchSpec('single_slice', False)]
# 
#     def concatenate_pipeline(self, **name_maps):  # @UnusedVariable
#         """
#         Concatenates two dMRI filesets (with different b-values) along the
#         DW encoding (4th) axis
#         """
# #             inputs=[FilesetSpec('low_b_dw_scan', mrtrix_image_format),
# #                     FilesetSpec('high_b_dw_scan', mrtrix_image_format)],
# #             outputs=[FilesetSpec('dwi_scan', mrtrix_image_format)],
#         pipeline = self.new_pipeline(
#             name='concatenation',
# 
#             desc=(
#                 "Concatenate low and high b-value dMRI filesets for NODDI "
#                 "processing"),
#             citations=[mrtrix_cite],
#             name_maps=name_maps)
#         # Create concatenation node
#         mrcat = pipeline.add('mrcat', MRCat(),
#                                      requirements=[mrtrix_req.v('3.0rc3')])
#                 quiet=True,  #  mrcat parameter
#         # Connect inputs
#             'first_scan': ('low_b_dw_scan', _format),  # input mrcat
#             'second_scan': ('high_b_dw_scan', _format),  # input mrcat
#         # Connect outputs
#         pipeline.connect_output('dwi_scan', mrcat, 'out_file')
#         # Check inputs/outputs are connected
#         return pipeline
# 
#     def noddi_fitting_pipeline(self, **name_maps):  # @UnusedVariable
#         """
#         Creates a ROI in which the NODDI processing will be performed
# 
#         Parameters
#         ----------
#         single_slice: Int
#             If provided the processing is only performed on a single slice
#             (for testing)
#         noddi_model: Str
#             Name of the NODDI model to use for the fitting
#         nthreads: Int
#             Number of processes to use
#         """
#         pipeline_name = 'noddi_fitting'
#         inputs = [FilesetSpec('bias_correct', nifti_gz_format),
#                   FilesetSpec('grad_dirs', fsl_bvecs_format),
#                   FilesetSpec('bvalues', fsl_bvals_format)]
#         if self.branch('single_slice'):
#             inputs.append(FilesetSpec('eroded_mask', nifti_gz_format))
#         else:
#             inputs.append(FilesetSpec('brain_mask', nifti_gz_format))
#         pipeline = self.new_pipeline(
#             name=pipeline_name,
#             inputs=inputs,
#             outputs=[FilesetSpec('ficvf', nifti_format),
#                      FilesetSpec('odi', nifti_format),
#                      FilesetSpec('fiso', nifti_format),
#                      FilesetSpec('fibredirs_xvec', nifti_format),
#                      FilesetSpec('fibredirs_yvec', nifti_format),
#                      FilesetSpec('fibredirs_zvec', nifti_format),
#                      FilesetSpec('fmin', nifti_format),
#                      FilesetSpec('kappa', nifti_format),
#                      FilesetSpec('error_code', nifti_format)],
#             desc=(
#                 "Creates a ROI in which the NODDI processing will be "
#                 "performed"),
#             citations=[noddi_cite],
#             name_maps=name_maps)
#         # Create node to unzip the nifti files
#         unzip_bias_correct = pipeline.add(
#             "unzip_bias_correct", MRConvert(),
#             requirements=[mrtrix_req.v('3.0rc3')])
#                 out_ext='nii',  #  unzip_bias_correct parameter
#                 quiet=True,  #  unzip_bias_correct parameter
#         unzip_mask = pipeline.add("unzip_mask", MRConvert(),
#                                   requirements=[mrtrix_req.v('3.0rc3')])
#                 out_ext='nii',  #  unzip_mask parameter
#                 quiet=True,  #  unzip_mask parameter
#         # Create create-roi node
#         create_roi = pipeline.add(
#             'create_roi',
#             CreateROI(),
#             requirements=[noddi_req, matlab_req.v('R2015a')],
#             mem_gb=4)
#             'in_file': (unzip_bias_correct, 'out_file'),  # internal create_roi
#             'brain_mask': (unzip_mask, 'out_file'),  # internal create_roi
#         # Create batch-fitting node
#         batch_fit = pipeline.add(
#             "batch_fit", BatchNODDIFitting(),
#             requirements=[noddi_req, matlab_req.v('R2015a')], wall_time=180,
#             mem_gb=8)
#                 model=self.parameter('noddi_model'),  #  batch_fit parameter
#                 nthreads=self.processor.num_processes,  #  batch_fit parameter
#             'roi_file': (create_roi, 'out_file'),  # internal batch_fit
#         # Create output node
#         save_params = pipeline.add(
#             "save_params",
#             SaveParamsAsNIfTI(),
#             requirements=[noddi_req, matlab_req.v('R2015a')],
#             mem_gb=4)
#                 output_prefix='params',  #  save_params parameter
#             'params_file': (batch_fit, 'out_file'),  # internal save_params
#             'roi_file': (create_roi, 'out_file'),  # internal save_params
#         pipeline.connect(unzip_mask, 'out_file', save_params,
#                          'brain_mask_file')
#         # Connect inputs
#             'in_file': ('bias_correct', _format),  # input unzip_bias_correct
#         if pipeline.branch('single_slice'):
#                 'in_file': ('brain_mask', _format),  # input unzip_mask
#         else:
#                 'in_file': ('eroded_mask', _format),  # input unzip_mask
#             'bvecs_file': ('grad_dirs', _format),  # input batch_fit
#             'bvals_file': ('bvalues', _format),  # input batch_fit
#         # Connect outputs
#         pipeline.connect_output('ficvf', save_params, 'ficvf')
#         pipeline.connect_output('odi', save_params, 'odi')
#         pipeline.connect_output('fiso', save_params, 'fiso')
#         pipeline.connect_output('fibredirs_xvec', save_params,
#                                 'fibredirs_xvec')
#         pipeline.connect_output('fibredirs_yvec', save_params,
#                                 'fibredirs_yvec')
#         pipeline.connect_output('fibredirs_zvec', save_params,
#                                 'fibredirs_zvec')
#         pipeline.connect_output('fmin', save_params, 'fmin')
#         pipeline.connect_output('kappa', save_params, 'kappa')
#         pipeline.connect_output('error_code', save_params, 'error_code')
#         # Check inputs/outputs are connected
#         return pipeline
