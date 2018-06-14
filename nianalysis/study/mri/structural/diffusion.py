from nipype.interfaces.utility import Merge
from nipype.interfaces.mrtrix3 import ResponseSD
from nipype.interfaces.mrtrix3.utils import BrainMask, TensorMetrics
from nipype.interfaces.mrtrix3.reconst import FitTensor, EstimateFOD
from nianalysis.interfaces.mrtrix import (
    DWIPreproc, MRCat, ExtractDWIorB0, MRMath, DWIBiasCorrect, DWIDenoise,
    MRCalc, DWIIntensityNorm, AverageResponse)
from nipype.workflows.dmri.fsl.tbss import create_tbss_all
from nianalysis.interfaces.noddi import (
    CreateROI, BatchNODDIFitting, SaveParamsAsNIfTI)
from nianalysis.interfaces.mrtrix import MRConvert, ExtractFSLGradients
from arcana.interfaces.utils import MergeTuple, Chain
from nipype.interfaces.utility import IdentityInterface
from nianalysis.citation import (
    mrtrix_cite, fsl_cite, eddy_cite, topup_cite, distort_correct_cite,
    noddi_cite, fast_cite, n4_cite, tbss_cite, dwidenoise_cites)
from nianalysis.file_format import (
    mrtrix_format, nifti_gz_format, fsl_bvecs_format, fsl_bvals_format,
    nifti_format, text_format, dicom_format, eddy_par_format, directory_format)
from nianalysis.requirement import (
    fsl509_req, mrtrix3_req, ants2_req, matlab2015_req, noddi_req, fsl510_req)
from arcana.study.base import StudyMetaClass
from arcana.dataset import DatasetSpec, FieldSpec
from arcana.interfaces.iterators import SelectSession
from arcana.parameter import ParameterSpec, SwitchSpec
from nianalysis.study.mri.epi import EPIStudy
from nipype.interfaces import fsl
from nianalysis.interfaces.custom.motion_correction import (
    PrepareDWI, AffineMatrixGeneration)


class DiffusionStudy(EPIStudy, metaclass=StudyMetaClass):

    add_data_specs = [
        DatasetSpec('dwi_reference', nifti_gz_format, optional=True),
        DatasetSpec('forward_pe', dicom_format, optional=True),
        DatasetSpec('b0', nifti_gz_format, 'extract_b0_pipeline',
                    desc="b0 image"),
        DatasetSpec('noise_residual', mrtrix_format, 'preproc_pipeline'),
        DatasetSpec('tensor', nifti_gz_format, 'tensor_pipeline'),
        DatasetSpec('fa', nifti_gz_format, 'tensor_pipeline'),
        DatasetSpec('adc', nifti_gz_format, 'tensor_pipeline'),
        DatasetSpec('wm_response', text_format, 'response_pipeline'),
        DatasetSpec('gm_response', text_format, 'response_pipeline'),
        DatasetSpec('csf_response', text_format, 'response_pipeline'),
        DatasetSpec('avg_response', text_format, 'average_response_pipeline'),
        DatasetSpec('fod', mrtrix_format, 'fod_pipeline'),
        DatasetSpec('bias_correct', nifti_gz_format, 'bias_correct_pipeline'),
        DatasetSpec('grad_dirs', fsl_bvecs_format, 'preproc_pipeline'),
        DatasetSpec('bvalues', fsl_bvals_format, 'preproc_pipeline'),
        DatasetSpec('eddy_par', eddy_par_format, 'preproc_pipeline'),
        DatasetSpec('align_mats', directory_format,
                    'intrascan_alignment_pipeline'),
        DatasetSpec('tbss_mean_fa', nifti_gz_format, 'tbss_pipeline',
                    frequency='per_project'),
        DatasetSpec('tbss_proj_fa', nifti_gz_format, 'tbss_pipeline',
                    frequency='per_project'),
        DatasetSpec('tbss_skeleton', nifti_gz_format, 'tbss_pipeline',
                    frequency='per_project'),
        DatasetSpec('tbss_skeleton_mask', nifti_gz_format, 'tbss_pipeline',
                    frequency='per_project'),
        DatasetSpec('brain', nifti_gz_format, 'brain_extraction_pipeline'),
        DatasetSpec('brain_mask', nifti_gz_format, 'brain_extraction_pipeline'),
        DatasetSpec('norm_intensity', mrtrix_format,
                    'intensity_normalisation_pipeline'),
        DatasetSpec('norm_intens_fa_template', mrtrix_format,
                    'intensity_normalisation_pipeline',
                    frequency='per_project'),
        DatasetSpec('norm_intens_wm_mask', mrtrix_format,
                    'intensity_normalisation_pipeline',
                    frequency='per_project')]

    add_parameter_specs = [
        ParameterSpec('multi_tissue', True),
        ParameterSpec('preproc_pe_dir', None, dtype=str),
        ParameterSpec('tbss_skel_thresh', 0.2),
        ParameterSpec('fsl_mask_f', 0.25),
        ParameterSpec('bet_robust', True),
        ParameterSpec('bet_f_threshold', 0.2),
        ParameterSpec('bet_reduce_bias', False)]

    add_switch_specs = [
        SwitchSpec('preproc_denoise', False),
        SwitchSpec('response_algorithm', 'tax',
                      ('tax', 'dhollander', 'msmt_5tt')),
        SwitchSpec('fod_algorithm', 'csd', ('csd', 'msmt_csd')),
        SwitchSpec('brain_extract_method', 'mrtrix',
                   ('mrtrix', 'fsl')),
        SwitchSpec('bias_correct_method', 'ants',
                   choices=('ants', 'fsl'))]

    def preproc_pipeline(self, **kwargs):  # @UnusedVariable @IgnorePep8
        """
        Performs a series of FSL preprocessing steps, including Eddy and Topup

        Parameters
        ----------
        phase_dir : str{AP|LR|IS}
            The phase encode direction
        """

        outputs = [DatasetSpec('preproc', nifti_gz_format),
                   DatasetSpec('grad_dirs', fsl_bvecs_format),
                   DatasetSpec('bvalues', fsl_bvals_format),
                   DatasetSpec('eddy_par', eddy_par_format)]
        citations = [fsl_cite, eddy_cite, topup_cite,
                     distort_correct_cite]
        if self.switch('preproc_denoise'):
            outputs.append(DatasetSpec('noise_residual', mrtrix_format))
            citations.extend(dwidenoise_cites)

        if 'dwi_reference' in self.input_names or 'reverse_phase' in self.input_names:
            inputs = [DatasetSpec('primary', dicom_format),
                      FieldSpec('ped', dtype=str),
                      FieldSpec('pe_angle', dtype=str)]
            if 'dwi_reference' in self.input_names:
                inputs.append(DatasetSpec('dwi_reference', nifti_gz_format))
            if 'reverse_phase' in self.input_names:
                inputs.append(DatasetSpec('reverse_phase', nifti_gz_format))
            distortion_correction = True
        else:
            inputs = [DatasetSpec('primary', dicom_format)]
            distortion_correction = False

        pipeline = self.create_pipeline(
            name='preprocess',
            inputs=inputs,
            outputs=outputs,
            desc=(
                "Preprocess dMRI studies using distortion correction"),
            version=1,
            citations=citations,
            **kwargs)
        # Denoise the dwi-scan
        if self.switch('preproc_denoise'):
            # Run denoising
            denoise = pipeline.create_node(DWIDenoise(), name='denoise',
                                           requirements=[mrtrix3_req])
            # Calculate residual noise
            subtract_operands = pipeline.create_node(Merge(2),
                                                     name='subtract_operands')
            subtract = pipeline.create_node(MRCalc(), name='subtract',
                                            requirements=[mrtrix3_req])
            subtract.inputs.operation = 'subtract'
        dwipreproc = pipeline.create_node(
            DWIPreproc(), name='dwipreproc',
            requirements=[mrtrix3_req, fsl510_req], wall_time=60)
        dwipreproc.inputs.eddy_parameters = '--data_is_shelled '
        dwipreproc.inputs.no_clean_up = True
        dwipreproc.inputs.out_file_ext = '.nii.gz'
        dwipreproc.inputs.temp_dir = 'dwipreproc_tempdir'
        # Create node to reorient preproc out_file
        swap = pipeline.create_node(
            fsl.utils.Reorient2Std(), name='fslreorient2std',
            requirements=[fsl509_req])
        if distortion_correction:
            # Extract b=0 volumes
            dwiextract = pipeline.create_node(
                ExtractDWIorB0(), name='dwiextract',
                requirements=[mrtrix3_req])
            dwiextract.inputs.bzero = True
            dwiextract.inputs.out_ext = '.nii.gz'
            # Get first b=0 from dwi b=0 volumes
            mrconvert = pipeline.create_node(MRConvert(), name="mrconvert",
                                             requirements=[mrtrix3_req])
            mrconvert.inputs.coord = (3, 0)
            # Concatenate extracted forward rpe with reverse rpe
            mrcat = pipeline.create_node(
                MRCat(), name='mrcat', requirements=[mrtrix3_req])
            # Create node to assign the right PED to the diffusion
            prep_dwi = pipeline.create_node(PrepareDWI(), name='prepare_dwi')
            # Create preprocessing node
            dwipreproc.inputs.rpe_pair = True
            if self.parameter('preproc_pe_dir') is not None:
                dwipreproc.inputs.pe_dir = self.parameter('preproc_pe_dir')
            # Create nodes to gradients to FSL format
            extract_grad = pipeline.create_node(
                ExtractFSLGradients(), name="extract_grad",
                requirements=[mrtrix3_req])
            # Connect inputs
            if 'dwi_reference' in self.input_names:
                pipeline.connect_input('dwi_reference', mrcat, 'second_scan')
            elif 'reverse_phase' in self.input_names:
                pipeline.connect_input('reverse_phase', mrcat, 'second_scan')
            else:
                assert False
            pipeline.connect_input('primary', dwiextract, 'in_file')
        # Connect inter-nodes
        if self.switch('preproc_denoise'):
            pipeline.connect_input('primary', denoise, 'in_file')
            pipeline.connect_input('primary', subtract_operands, 'in1')
            pipeline.connect(denoise, 'out_file', dwipreproc, 'in_file')
            pipeline.connect(denoise, 'noise', subtract_operands, 'in2')
            pipeline.connect(subtract_operands, 'out', subtract, 'operands')
        else:
            pipeline.connect_input('primary', dwipreproc, 'in_file')
        if distortion_correction:
            pipeline.connect_input('ped', prep_dwi, 'pe_dir')
            pipeline.connect_input('pe_angle', prep_dwi, 'ped_polarity')
            pipeline.connect(prep_dwi, 'pe', dwipreproc, 'pe_dir')
            pipeline.connect(mrcat, 'out_file', dwipreproc, 'se_epi')
            pipeline.connect(dwiextract, 'out_file', mrconvert, 'in_file')
            pipeline.connect(mrconvert, 'out_file', mrcat, 'first_scan')
        pipeline.connect_input('primary', extract_grad, 'in_file')
        pipeline.connect(dwipreproc, 'out_file', swap, 'in_file')
        # Connect outputs
        pipeline.connect_output('preproc', swap, 'out_file')
        pipeline.connect_output('grad_dirs', extract_grad,
                                'bvecs_file')
        pipeline.connect_output('bvalues', extract_grad, 'bvals_file')
        pipeline.connect_output('eddy_par', dwipreproc, 'eddy_parameters')
        if self.switch('preproc_denoise'):
            pipeline.connect_output('noise_residual', subtract, 'out_file')
        # Check inputs/outputs are connected
        return pipeline

    def brain_extraction_pipeline(self, **kwargs):  # @UnusedVariable @IgnorePep8
        """
        Generates a whole brain mask using MRtrix's 'dwi2mask' command

        Parameters
        ----------
        mask_tool: Str
            Can be either 'bet' or 'dwi2mask' depending on which mask tool you
            want to use
        """
        if self.branch('brain_extract_method', 'mrtrix'):
            pipeline = self.create_pipeline(
                'brain_extraction',
                inputs=[DatasetSpec('preproc', nifti_gz_format),
                        DatasetSpec('grad_dirs', fsl_bvecs_format),
                        DatasetSpec('bvalues', fsl_bvals_format)],
                outputs=[DatasetSpec('brain_mask', nifti_gz_format)],
                desc="Generate brain mask from b0 images",
                version=1,
                citations=[mrtrix_cite],
                **kwargs)
            # Create mask node
            dwi2mask = pipeline.create_node(BrainMask(), name='dwi2mask',
                                            requirements=[mrtrix3_req])
            dwi2mask.inputs.out_file = 'brain_mask.nii.gz'
            # Gradient merge node
            grad_fsl = pipeline.create_node(MergeTuple(2), name="grad_fsl")
            # Connect nodes
            pipeline.connect(grad_fsl, 'out', dwi2mask, 'grad_fsl')
            # Connect inputs
            pipeline.connect_input('grad_dirs', grad_fsl, 'in1')
            pipeline.connect_input('bvalues', grad_fsl, 'in2')
            pipeline.connect_input('preproc', dwi2mask, 'in_file')
            # Connect outputs
            pipeline.connect_output('brain_mask', dwi2mask, 'out_file')
            # Check inputs/outputs are connected
            pipeline.assert_connected()
        else:
            pipeline = super(DiffusionStudy, self).brain_extraction_pipeline(
                **kwargs)
        return pipeline

    def bias_correct_pipeline(self, **kwargs):  # @UnusedVariable @IgnorePep8
        """
        Corrects B1 field inhomogeneities
        """
        bias_method = self.switch('bias_correct_method')
        pipeline = self.create_pipeline(
            name='bias_correct',
            inputs=[DatasetSpec('preproc', nifti_gz_format),
                    DatasetSpec('brain_mask', nifti_gz_format),
                    DatasetSpec('grad_dirs', fsl_bvecs_format),
                    DatasetSpec('bvalues', fsl_bvals_format)],
            outputs=[DatasetSpec('bias_correct', nifti_gz_format)],
            desc="Corrects for B1 field inhomogeneity",
            version=1,
            citations=[fast_cite,
                       (n4_cite if bias_method == 'ants' else fsl_cite)],
            **kwargs)
        # Create bias correct node
        bias_correct = pipeline.create_node(
            DWIBiasCorrect(), name="bias_correct",
            requirements=(
                [mrtrix3_req] +
                [ants2_req if bias_method == 'ants' else fsl509_req]))
        bias_correct.inputs.method = bias_method
        # Gradient merge node
        fsl_grads = pipeline.create_node(MergeTuple(2), name="fsl_grads")
        # Connect nodes
        pipeline.connect(fsl_grads, 'out', bias_correct, 'grad_fsl')
        # Connect to inputs
        pipeline.connect_input('grad_dirs', fsl_grads, 'in1')
        pipeline.connect_input('bvalues', fsl_grads, 'in2')
        pipeline.connect_input('preproc', bias_correct, 'in_file')
        pipeline.connect_input('brain_mask', bias_correct, 'mask')
        # Connect to outputs
        pipeline.connect_output('bias_correct', bias_correct, 'out_file')
        # Check inputs/output are connected
        return pipeline

    def intensity_normalisation_pipeline(self, **kwargs):
        pipeline = self.create_pipeline(
            name='intensity_normalization',
            inputs=[DatasetSpec('bias_correct', nifti_gz_format),
                    DatasetSpec('brain_mask', nifti_gz_format),
                    DatasetSpec('grad_dirs', fsl_bvecs_format),
                    DatasetSpec('bvalues', fsl_bvals_format)],
            outputs=[DatasetSpec('norm_intensity', mrtrix_format),
                     DatasetSpec('norm_intens_fa_template', mrtrix_format,
                                 frequency='per_project'),
                     DatasetSpec('norm_intens_wm_mask', mrtrix_format,
                                 frequency='per_project')],
            desc="Corrects for B1 field inhomogeneity",
            version=1,
            citations=[mrtrix3_req],
            **kwargs)
        # Convert from nifti to mrtrix format
        grad_merge = pipeline.create_node(MergeTuple(2), name="grad_merge")
        mrconvert = pipeline.create_node(MRConvert(), name='mrconvert')
        mrconvert.inputs.out_ext = '.mif'
        # Set up join nodes
        fields = ['dwis', 'masks', 'subject_ids', 'visit_ids']
        join_subjects = pipeline.create_join_subjects_node(
            IdentityInterface(fields), joinfield=fields,
            name='join_subjects')
        join_visits = pipeline.create_join_visits_node(
            Chain(fields), joinfield=fields,
            name='join_visits')
        # Set up expand nodes
        select = pipeline.create_node(
            SelectSession(), name='expand')
        # Intensity normalization
        intensity_norm = pipeline.create_node(
            DWIIntensityNorm(), name='dwiintensitynorm')
        # Connect inputs
        pipeline.connect_input('bias_correct', mrconvert, 'in_file')
        pipeline.connect_input('grad_dirs', grad_merge, 'in1')
        pipeline.connect_input('bvalues', grad_merge, 'in2')
        pipeline.connect_subject_id(join_subjects, 'subject_ids')
        pipeline.connect_visit_id(join_subjects, 'visit_ids')
        pipeline.connect_subject_id(select, 'subject_id')
        pipeline.connect_visit_id(select, 'visit_id')
        pipeline.connect_input('brain_mask', join_subjects, 'masks')
        # Internal connections
        pipeline.connect(grad_merge, 'out', mrconvert, 'grad_fsl')
        pipeline.connect(mrconvert, 'out_file', join_subjects, 'dwis')
        pipeline.connect(join_subjects, 'dwis', join_visits, 'dwis')
        pipeline.connect(join_subjects, 'masks', join_visits, 'masks')
        pipeline.connect(join_subjects, 'subject_ids', join_visits,
                         'subject_ids')
        pipeline.connect(join_subjects, 'visit_ids', join_visits,
                         'visit_ids')
        pipeline.connect(join_visits, 'dwis', intensity_norm, 'in_files')
        pipeline.connect(join_visits, 'masks', intensity_norm, 'masks')
        pipeline.connect(join_visits, 'subject_ids', select, 'subject_ids')
        pipeline.connect(join_visits, 'visit_ids', select, 'visit_ids')
        pipeline.connect(intensity_norm, 'out_files', select, 'items')
        # Connect outputs
        pipeline.connect_output('norm_intensity', select, 'item')
        pipeline.connect_output('norm_intens_fa_template', intensity_norm,
                                'fa_template')
        pipeline.connect_output('norm_intens_wm_mask', intensity_norm,
                                'wm_mask')
        return pipeline

    def tensor_pipeline(self, **kwargs):  # @UnusedVariable
        """
        Fits the apparrent diffusion tensor (DT) to each voxel of the image
        """
        pipeline = self.create_pipeline(
            name='tensor',
            inputs=[DatasetSpec('bias_correct', nifti_gz_format),
                    DatasetSpec('grad_dirs', fsl_bvecs_format),
                    DatasetSpec('bvalues', fsl_bvals_format),
                    DatasetSpec('brain_mask', nifti_gz_format)],
            outputs=[DatasetSpec('tensor', nifti_gz_format)],
            desc=("Estimates the apparent diffusion tensor in each "
                  "voxel"),
            version=1,
            citations=[],
            **kwargs)
        # Create tensor fit node
        dwi2tensor = pipeline.create_node(FitTensor(), name='dwi2tensor')
        dwi2tensor.inputs.out_file = 'dti.nii.gz'
        # Gradient merge node
        fsl_grads = pipeline.create_node(MergeTuple(2), name="fsl_grads")
        # Connect nodes
        pipeline.connect(fsl_grads, 'out', dwi2tensor, 'grad_fsl')
        # Connect to inputs
        pipeline.connect_input('grad_dirs', fsl_grads, 'in1')
        pipeline.connect_input('bvalues', fsl_grads, 'in2')
        pipeline.connect_input('bias_correct', dwi2tensor, 'in_file')
        pipeline.connect_input('brain_mask', dwi2tensor, 'in_mask')
        # Connect to outputs
        pipeline.connect_output('tensor', dwi2tensor, 'out_file')
        # Check inputs/output are connected
        return pipeline

    def fa_pipeline(self, **kwargs):  # @UnusedVariable
        """
        Fits the apparrent diffusion tensor (DT) to each voxel of the image
        """
        pipeline = self.create_pipeline(
            name='fa',
            inputs=[DatasetSpec('tensor', nifti_gz_format),
                    DatasetSpec('brain_mask', nifti_gz_format)],
            outputs=[DatasetSpec('fa', nifti_gz_format),
                     DatasetSpec('adc', nifti_gz_format)],
            desc=("Calculates the FA and ADC from a tensor image"),
            version=1,
            citations=[],
            **kwargs)
        # Create tensor fit node
        metrics = pipeline.create_node(TensorMetrics(), name='metrics',
                                       requirements=[mrtrix3_req])
        metrics.inputs.out_fa = 'fa.nii.gz'
        metrics.inputs.out_adc = 'adc.nii.gz'
        # Connect to inputs
        pipeline.connect_input('tensor', metrics, 'in_file')
        pipeline.connect_input('brain_mask', metrics, 'in_mask')
        # Connect to outputs
        pipeline.connect_output('fa', metrics, 'out_fa')
        pipeline.connect_output('adc', metrics, 'out_adc')
        # Check inputs/output are connected
        return pipeline

    def response_pipeline(self, **kwargs):  # @UnusedVariable
        """
        Estimates the fibre orientation distribution (FOD) using constrained
        spherical deconvolution

        Parameters
        ----------
        response_algorithm : str
            Algorithm used to estimate the response
        """
        outputs = [DatasetSpec('wm_response', text_format)]
        if self.branch('response_algorithm', ('dhollander', 'msmt_5tt')):
            outputs.append(DatasetSpec('gm_response', text_format))
            outputs.append(DatasetSpec('csf_response', text_format))
        pipeline = self.create_pipeline(
            name='response',
            inputs=[DatasetSpec('bias_correct', nifti_gz_format),
                    DatasetSpec('grad_dirs', fsl_bvecs_format),
                    DatasetSpec('bvalues', fsl_bvals_format),
                    DatasetSpec('brain_mask', nifti_gz_format)],
            outputs=outputs,
            desc=("Estimates the fibre response function"),
            version=1,
            citations=[mrtrix_cite],
            **kwargs)
        # Create fod fit node
        response = pipeline.create_node(ResponseSD(), name='response',
                                        requirements=[mrtrix3_req])
        response.inputs.algorithm = self.parameter('response_algorithm')
        # Gradient merge node
        fsl_grads = pipeline.create_node(MergeTuple(2), name="fsl_grads")
        # Connect nodes
        pipeline.connect(fsl_grads, 'out', response, 'grad_fsl')
        # Connect to inputs
        pipeline.connect_input('grad_dirs', fsl_grads, 'in1')
        pipeline.connect_input('bvalues', fsl_grads, 'in2')
        pipeline.connect_input('bias_correct', response, 'in_file')
        pipeline.connect_input('brain_mask', response, 'in_mask')
        # Connect to outputs
        pipeline.connect_output('wm_response', response, 'wm_file')
        if self.branch('response_algorithm', ('dhollander', 'msmt_5tt')):
            pipeline.connect_output('gm_response', response, 'gm_file')
            pipeline.connect_output('csf_response', response, 'csf_file')
        # Check inputs/output are connected
        return pipeline

    def average_response_pipeline(self, **kwargs):
        """
        Averages the estimate response function over all subjects in the
        project
        """
        pipeline = self.create_pipeline(
            name='average_response',
            inputs=[DatasetSpec('wm_response', text_format)],
            outputs=[DatasetSpec('avg_response', text_format,
                                 frequency='per_project')],
            desc=(
                "Averages the fibre response function over the project"),
            version=1,
            citations=[mrtrix_cite],
            **kwargs)
        join_subjects = pipeline.create_join_subjects_node(
            IdentityInterface(['responses']), name='join_subjects',
            joinfield=['responses'])
        join_visits = pipeline.create_join_visits_node(
            Chain(['responses']), name='join_visits', joinfield=['responses'])
        avg_response = pipeline.create_node(AverageResponse(),
                                            name='avg_response')
        # Connect inputs
        pipeline.connect_input('wm_response', join_subjects, 'responses')
        # Connect inter-nodes
        pipeline.connect(join_subjects, 'responses', join_visits, 'responses')
        pipeline.connect(join_visits, 'responses', avg_response, 'in_files')
        # Connect outputs
        pipeline.connect_output('avg_response', avg_response, 'out_file')
        # Check inputs/output are connected
        return pipeline
# 
# algorithm = traits.Enum(
#         'csd',
#         'msmt_csd',
#         argstr='%s',
#         position=-8,
#         mandatory=True,
#         desc='FOD algorithm')
#     in_file = File(
#         exists=True,
#         argstr='%s',
#         position=-7,
#         mandatory=True,
#         desc='input DWI image')
#     wm_txt = File(
#         argstr='%s', position=-6, mandatory=True, desc='WM response text file')
#     wm_odf = File(
#         'wm.mif',
#         argstr='%s',
#         position=-5,
#         usedefault=True,
#         mandatory=True,
#         desc='output WM ODF')
#     gm_txt = File(argstr='%s', position=-4, desc='GM response text file')
#     gm_odf = File('gm.mif', usedefault=True, argstr='%s',
#                   position=-3, desc='output GM ODF')
#     csf_txt = File(argstr='%s', position=-2, desc='CSF response text file')
#     csf_odf = File('csf.mif', usedefault=True, argstr='%s',
#                    position=-1, desc='output CSF ODF')
#     mask_file = File(exists=True, argstr='-mask %s', desc='mask image')

    def fod_pipeline(self, **kwargs):  # @UnusedVariable
        """
        Estimates the fibre orientation distribution (FOD) using constrained
        spherical deconvolution

        Parameters
        ----------
        """
        pipeline = self.create_pipeline(
            name='fod',
            inputs=[DatasetSpec('bias_correct', nifti_gz_format),
                    DatasetSpec('grad_dirs', fsl_bvecs_format),
                    DatasetSpec('bvalues', fsl_bvals_format),
                    DatasetSpec('wm_response', text_format),
                    DatasetSpec('brain_mask', nifti_gz_format)],
            outputs=[DatasetSpec('fod', nifti_gz_format)],
            desc=("Estimates the fibre orientation distribution in each"
                  " voxel"),
            version=1,
            citations=[mrtrix_cite],
            **kwargs)
        if self.branch('fod_algorithm', 'msmt_csd'):
            pipeline.add_input(DatasetSpec('gm_response', text_format))
            pipeline.add_input(DatasetSpec('csf_response', text_format))
        # Create fod fit node
        dwi2fod = pipeline.create_node(EstimateFOD(), name='dwi2fod',
                                       requirements=[mrtrix3_req])
        dwi2fod.inputs.algorithm = self.switch('fod_algorithm')
        # Gradient merge node
        fsl_grads = pipeline.create_node(MergeTuple(2), name="fsl_grads")
        # Connect nodes
        pipeline.connect(fsl_grads, 'out', dwi2fod, 'grad_fsl')
        # Connect to inputs
        pipeline.connect_input('grad_dirs', fsl_grads, 'in1')
        pipeline.connect_input('bvalues', fsl_grads, 'in2')
        pipeline.connect_input('bias_correct', dwi2fod, 'in_file')
        pipeline.connect_input('wm_response', dwi2fod, 'wm_txt')
        pipeline.connect_input('brain_mask', dwi2fod, 'mask_file')
        # Connect to outputs
        pipeline.connect_output('fod', dwi2fod, 'wm_odf')
        # Check inputs/output are connected
        return pipeline

    def tbss_pipeline(self, **kwargs):  # @UnusedVariable
        pipeline = self.create_pipeline(
            name='tbss',
            inputs=[DatasetSpec('fa', nifti_gz_format)],
            outputs=[DatasetSpec('tbss_mean_fa', nifti_gz_format),
                     DatasetSpec('tbss_proj_fa', nifti_gz_format,
                                 frequency='per_project'),
                     DatasetSpec('tbss_skeleton', nifti_gz_format,
                                 frequency='per_project'),
                     DatasetSpec('tbss_skeleton_mask', nifti_gz_format,
                                 frequency='per_project')],
            version=1,
            citations=[tbss_cite, fsl_cite],
            **kwargs)
        # Create TBSS workflow
        tbss = create_tbss_all(name='tbss')
        # Connect inputs
        pipeline.connect_input('fa', tbss, 'inputnode.fa_list')
        # Connect outputs
        pipeline.connect_output('tbss_mean_fa', tbss,
                                'outputnode.meanfa_file')
        pipeline.connect_output('tbss_proj_fa', tbss,
                                'outputnode.projectedfa_file')
        pipeline.connect_output('tbss_skeleton', tbss,
                                'outputnode.skeleton_file')
        pipeline.connect_output('tbss_skeleton_mask', tbss,
                                'outputnode.skeleton_mask')
        # Check inputs/output are connected
        return pipeline

    def extract_b0_pipeline(self, **kwargs):  # @UnusedVariable
        """
        Extracts the b0 images from a DWI study and takes their mean
        """
        pipeline = self.create_pipeline(
            name='extract_b0',
            inputs=[DatasetSpec('bias_correct', nifti_gz_format),
                    DatasetSpec('grad_dirs', fsl_bvecs_format),
                    DatasetSpec('bvalues', fsl_bvals_format)],
            outputs=[DatasetSpec('b0', nifti_gz_format)],
            desc="Extract b0 image from a DWI study",
            version=1,
            citations=[mrtrix_cite],
            **kwargs)
        # Gradient merge node
        fsl_grads = pipeline.create_node(MergeTuple(2), name="fsl_grads")
        # Extraction node
        extract_b0s = pipeline.create_node(
            ExtractDWIorB0(), name='extract_b0s',
            requirements=[mrtrix3_req])
        extract_b0s.inputs.bzero = True
        extract_b0s.inputs.quiet = True
        # FIXME: Need a registration step before the mean
        # Mean calculation node
        mean = pipeline.create_node(MRMath(), name="mean",
                                    requirements=[mrtrix3_req])
        mean.inputs.axis = 3
        mean.inputs.operation = 'mean'
        mean.inputs.quiet = True
        # Convert to Nifti
        mrconvert = pipeline.create_node(MRConvert(), name="output_conversion",
                                         requirements=[mrtrix3_req])
        mrconvert.inputs.out_ext = '.nii.gz'
        mrconvert.inputs.quiet = True
        # Connect inputs
        pipeline.connect_input('bias_correct', extract_b0s, 'in_file')
        pipeline.connect_input('grad_dirs', fsl_grads, 'in1')
        pipeline.connect_input('bvalues', fsl_grads, 'in2')
        # Connect between nodes
        pipeline.connect(extract_b0s, 'out_file', mean, 'in_files')
        pipeline.connect(fsl_grads, 'out', extract_b0s, 'grad_fsl')
        pipeline.connect(mean, 'out_file', mrconvert, 'in_file')
        # Connect outputs
        pipeline.connect_output('b0', mrconvert, 'out_file')
        pipeline.assert_connected()
        # Check inputs/outputs are connected
        return pipeline

    def track_gen_pipeline(self, **kwargs):
        pipeline = self.create_pipeline(
            name='extract_b0',
            inputs=[DatasetSpec('bias_correct', nifti_gz_format),
                    DatasetSpec('grad_dirs', fsl_bvecs_format),
                    DatasetSpec('bvalues', fsl_bvals_format)],
            outputs=[DatasetSpec('b0', nifti_gz_format)],
            desc="Extract b0 image from a DWI study",
            version=1,
            citations=[mrtrix_cite])
        return pipeline

    def intrascan_alignment_pipeline(self, **kwargs):

        pipeline = self.create_pipeline(
            name='affine_mat_generation',
            inputs=[DatasetSpec('preproc', nifti_gz_format),
                    DatasetSpec('eddy_par', eddy_par_format)],
            outputs=[
                DatasetSpec('align_mats', directory_format)],
            desc=("Generation of the affine matrices for the main dwi "
                  "sequence starting from eddy motion parameters"),
            version=1,
            citations=[fsl_cite],
            **kwargs)

        aff_mat = pipeline.create_node(AffineMatrixGeneration(),
                                       name='gen_aff_mats')
        pipeline.connect_input('preproc', aff_mat, 'reference_image')
        pipeline.connect_input(
            'eddy_par', aff_mat, 'motion_parameters')
        pipeline.connect_output(
            'align_mats', aff_mat, 'affine_matrices')
        return pipeline


class NODDIStudy(DiffusionStudy, metaclass=StudyMetaClass):

    add_data_specs = [
        DatasetSpec('low_b_dw_scan', mrtrix_format),
        DatasetSpec('high_b_dw_scan', mrtrix_format),
        DatasetSpec('forward_pe', mrtrix_format),
        DatasetSpec('reverse_pe', mrtrix_format),
        DatasetSpec('dwi_scan', mrtrix_format, 'concatenate_pipeline'),
        DatasetSpec('ficvf', nifti_format, 'noddi_fitting_pipeline'),
        DatasetSpec('odi', nifti_format, 'noddi_fitting_pipeline'),
        DatasetSpec('fiso', nifti_format, 'noddi_fitting_pipeline'),
        DatasetSpec('fibredirs_xvec', nifti_format, 'noddi_fitting_pipeline'),
        DatasetSpec('fibredirs_yvec', nifti_format, 'noddi_fitting_pipeline'),
        DatasetSpec('fibredirs_zvec', nifti_format, 'noddi_fitting_pipeline'),
        DatasetSpec('fmin', nifti_format, 'noddi_fitting_pipeline'),
        DatasetSpec('kappa', nifti_format, 'noddi_fitting_pipeline'),
        DatasetSpec('error_code', nifti_format, 'noddi_fitting_pipeline')]

    add_parameter_specs = [ParameterSpec('noddi_model',
                                   'WatsonSHStickTortIsoV_B0')]

    add_switch_specs = [
        SwitchSpec('single_slice', False)]

    def concatenate_pipeline(self, **kwargs):  # @UnusedVariable
        """
        Concatenates two dMRI datasets (with different b-values) along the
        DW encoding (4th) axis
        """
        pipeline = self.create_pipeline(
            name='concatenation',
            inputs=[DatasetSpec('low_b_dw_scan', mrtrix_format),
                    DatasetSpec('high_b_dw_scan', mrtrix_format)],
            outputs=[DatasetSpec('dwi_scan', mrtrix_format)],
            desc=(
                "Concatenate low and high b-value dMRI datasets for NODDI "
                "processing"),
            version=1,
            citations=[mrtrix_cite],
            **kwargs)
        # Create concatenation node
        mrcat = pipeline.create_node(MRCat(), name='mrcat',
                                     requirements=[mrtrix3_req])
        mrcat.inputs.quiet = True
        # Connect inputs
        pipeline.connect_input('low_b_dw_scan', mrcat, 'first_scan')
        pipeline.connect_input('high_b_dw_scan', mrcat, 'second_scan')
        # Connect outputs
        pipeline.connect_output('dwi_scan', mrcat, 'out_file')
        # Check inputs/outputs are connected
        return pipeline

    def noddi_fitting_pipeline(self, **kwargs):  # @UnusedVariable
        """
        Creates a ROI in which the NODDI processing will be performed

        Parameters
        ----------
        single_slice: Int
            If provided the processing is only performed on a single slice
            (for testing)
        noddi_model: Str
            Name of the NODDI model to use for the fitting
        nthreads: Int
            Number of processes to use
        """
        pipeline_name = 'noddi_fitting'
        inputs = [DatasetSpec('bias_correct', nifti_gz_format),
                  DatasetSpec('grad_dirs', fsl_bvecs_format),
                  DatasetSpec('bvalues', fsl_bvals_format)]
        if self.switch('single_slice'):
            inputs.append(DatasetSpec('eroded_mask', nifti_gz_format))
        else:
            inputs.append(DatasetSpec('brain_mask', nifti_gz_format))
        pipeline = self.create_pipeline(
            name=pipeline_name,
            inputs=inputs,
            outputs=[DatasetSpec('ficvf', nifti_format),
                     DatasetSpec('odi', nifti_format),
                     DatasetSpec('fiso', nifti_format),
                     DatasetSpec('fibredirs_xvec', nifti_format),
                     DatasetSpec('fibredirs_yvec', nifti_format),
                     DatasetSpec('fibredirs_zvec', nifti_format),
                     DatasetSpec('fmin', nifti_format),
                     DatasetSpec('kappa', nifti_format),
                     DatasetSpec('error_code', nifti_format)],
            desc=(
                "Creates a ROI in which the NODDI processing will be "
                "performed"),
            citations=[noddi_cite],
            **kwargs)
        # Create node to unzip the nifti files
        unzip_bias_correct = pipeline.create_node(
            MRConvert(), name="unzip_bias_correct",
            requirements=[mrtrix3_req])
        unzip_bias_correct.inputs.out_ext = 'nii'
        unzip_bias_correct.inputs.quiet = True
        unzip_mask = pipeline.create_node(MRConvert(), name="unzip_mask",
                                          requirements=[mrtrix3_req])
        unzip_mask.inputs.out_ext = 'nii'
        unzip_mask.inputs.quiet = True
        # Create create-roi node
        create_roi = pipeline.create_node(
            CreateROI(), name='create_roi',
            requirements=[noddi_req, matlab2015_req],
            memory=4000)
        pipeline.connect(unzip_bias_correct, 'out_file', create_roi, 'in_file')
        pipeline.connect(unzip_mask, 'out_file', create_roi, 'brain_mask')
        # Create batch-fitting node
        batch_fit = pipeline.create_node(
            BatchNODDIFitting(), name="batch_fit",
            requirements=[noddi_req, matlab2015_req], wall_time=180,
            memory=8000)
        batch_fit.inputs.model = self.parameter('noddi_model')
        batch_fit.inputs.nthreads = self.runner.num_processes
        pipeline.connect(create_roi, 'out_file', batch_fit, 'roi_file')
        # Create output node
        save_params = pipeline.create_node(
            SaveParamsAsNIfTI(), name="save_params",
            requirements=[noddi_req, matlab2015_req],
            memory=4000)
        save_params.inputs.output_prefix = 'params'
        pipeline.connect(batch_fit, 'out_file', save_params, 'params_file')
        pipeline.connect(create_roi, 'out_file', save_params, 'roi_file')
        pipeline.connect(unzip_mask, 'out_file', save_params,
                         'brain_mask_file')
        # Connect inputs
        pipeline.connect_input('bias_correct', unzip_bias_correct, 'in_file')
        if not pipeline.switch('single_slice'):
            pipeline.connect_input('eroded_mask', unzip_mask, 'in_file')
        else:
            pipeline.connect_input('brain_mask', unzip_mask, 'in_file')
        pipeline.connect_input('grad_dirs', batch_fit, 'bvecs_file')
        pipeline.connect_input('bvalues', batch_fit, 'bvals_file')
        # Connect outputs
        pipeline.connect_output('ficvf', save_params, 'ficvf')
        pipeline.connect_output('odi', save_params, 'odi')
        pipeline.connect_output('fiso', save_params, 'fiso')
        pipeline.connect_output('fibredirs_xvec', save_params,
                                'fibredirs_xvec')
        pipeline.connect_output('fibredirs_yvec', save_params,
                                'fibredirs_yvec')
        pipeline.connect_output('fibredirs_zvec', save_params,
                                'fibredirs_zvec')
        pipeline.connect_output('fmin', save_params, 'fmin')
        pipeline.connect_output('kappa', save_params, 'kappa')
        pipeline.connect_output('error_code', save_params, 'error_code')
        # Check inputs/outputs are connected
        return pipeline
