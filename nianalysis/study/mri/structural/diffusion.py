from nipype.pipeline import engine as pe
from nipype.interfaces.mrtrix3.utils import BrainMask, TensorMetrics
from nipype.interfaces.mrtrix3.reconst import FitTensor, EstimateFOD
from nipype.interfaces.mrtrix3.preprocess import ResponseSD
from nianalysis.interfaces.mrtrix import (
    DWIPreproc, MRCat, ExtractDWIorB0, MRMath, DWIBiasCorrect)
from nipype.workflows.dmri.fsl.tbss import create_tbss_all
from nianalysis.interfaces.noddi import (
    CreateROI, BatchNODDIFitting, SaveParamsAsNIfTI)
from .t2 import T2Study
from nianalysis.interfaces.mrtrix import MRConvert, ExtractFSLGradients
from nianalysis.interfaces.utils import MergeTuple
from nianalysis.citations import (
    mrtrix_cite, fsl_cite, eddy_cite, topup_cite, distort_correct_cite,
    noddi_cite, fast_cite, n4_cite, tbss_cite)
from nianalysis.data_formats import (
    mrtrix_format, nifti_gz_format, fsl_bvecs_format, fsl_bvals_format,
    nifti_format)
from nianalysis.requirements import (
    fsl5_req, mrtrix3_req, Requirement, ants2_req)
from nianalysis.exceptions import NiAnalysisError
from nianalysis.study.base import set_dataset_specs
from nianalysis.dataset import DatasetSpec


class DiffusionStudy(T2Study):

    def preprocess_pipeline(self, **options):  # @UnusedVariable @IgnorePep8
        """
        Performs a series of FSL preprocessing steps, including Eddy and Topup

        Parameters
        ----------
        phase_dir : str{AP|LR|IS}
            The phase encode direction
        """
        pipeline = self.create_pipeline(
            name='preprocess',
            inputs=[DatasetSpec('dwi_scan', mrtrix_format),
                    DatasetSpec('forward_rpe', mrtrix_format),
                    DatasetSpec('reverse_rpe', mrtrix_format)],
            outputs=[DatasetSpec('dwi_preproc', nifti_gz_format),
                     DatasetSpec('grad_dirs', fsl_bvecs_format),
                     DatasetSpec('bvalues', fsl_bvals_format)],
            description="Preprocess dMRI studies using distortion correction",
            default_options={'phase_dir': 'LR'},
            version=1,
            requirements=[mrtrix3_req, fsl5_req],
            citations=[fsl_cite, eddy_cite, topup_cite, distort_correct_cite],
            approx_runtime=30, options=options)
        # Create preprocessing node
        dwipreproc = pipeline.create_node(DWIPreproc(), name='dwipreproc')
        dwipreproc.inputs.pe_dir = pipeline.option('phase_dir')
        # Create nodes to convert preprocessed dataset and gradients to FSL
        # format
        mrconvert = pipeline.create_node(MRConvert(), name='mrconvert')
        mrconvert.inputs.out_ext = '.nii.gz'
        mrconvert.inputs.quiet = True
        extract_grad = pipeline.create_node(ExtractFSLGradients(), name="extract_grad")
        pipeline.connect(dwipreproc, 'out_file', mrconvert, 'in_file')
        pipeline.connect(dwipreproc, 'out_file', extract_grad, 'in_file')
        # Connect inputs
        pipeline.connect_input('dwi_scan', dwipreproc, 'in_file')
        pipeline.connect_input('forward_rpe', dwipreproc, 'forward_rpe')
        pipeline.connect_input('reverse_rpe', dwipreproc, 'reverse_rpe')
        # Connect outputs
        pipeline.connect_output('dwi_preproc', mrconvert, 'out_file')
        pipeline.connect_output('grad_dirs', extract_grad,
                                'bvecs_file')
        pipeline.connect_output('bvalues', extract_grad, 'bvals_file')
        # Check inputs/outputs are connected
        pipeline.assert_connected()
        return pipeline

    def brain_mask_pipeline(self, mask_tool='mrtrix', **options):  # @UnusedVariable @IgnorePep8
        """
        Generates a whole brain mask using MRtrix's 'dwi2mask' command

        Parameters
        ----------
        mask_tool: Str
            Can be either 'bet' or 'dwi2mask' depending on which mask tool you
            want to use
        """
        if mask_tool == 'fsl':
            if 'f' not in options:
                options['f'] = 0.25
            pipeline = super(DiffusionStudy, self).brain_mask_pipeline(
                **options)
        elif mask_tool == 'mrtrix':
            pipeline = self.create_pipeline(
                name='brain_mask_mrtrix',
                inputs=[DatasetSpec('dwi_preproc', nifti_gz_format),
                        DatasetSpec('grad_dirs', fsl_bvecs_format),
                        DatasetSpec('bvalues', fsl_bvals_format)],
                outputs=[DatasetSpec('brain_mask', nifti_gz_format)],
                description="Generate brain mask from b0 images",
                default_options={},
                version=1,
                requirements=[mrtrix3_req],
                citations=[mrtrix_cite], approx_runtime=1,
                options=options)
            # Create mask node
            dwi2mask = pipeline.create_node(BrainMask(), name='dwi2mask')
            dwi2mask.inputs.out_file = 'brain_mask.nii.gz'
            # Gradient merge node
            grad_fsl = pipeline.create_node(MergeTuple(2), name="grad_fsl")
            # Connect nodes
            pipeline.connect(grad_fsl, 'out', dwi2mask, 'grad_fsl')
            # Connect inputs
            pipeline.connect_input('grad_dirs', grad_fsl, 'in1')
            pipeline.connect_input('bvalues', grad_fsl, 'in2')
            pipeline.connect_input('dwi_preproc', dwi2mask, 'in_file')
            # Connect outputs
            pipeline.connect_output('brain_mask', dwi2mask, 'out_file')
            # Check inputs/outputs are connected
            pipeline.assert_connected()
        else:
            raise NiAnalysisError(
                "Unrecognised mask_tool '{}' (valid options 'bet' or "
                "'dwi2mask')")
        return pipeline

    def bias_correct_pipeline(self, **options):  # @UnusedVariable @IgnorePep8
        """
        Corrects B1 field inhomogeneities
        """
        bias_method_default = 'ants'
        bias_method = options.get('bias_method', bias_method_default)
        if bias_method not in ('ants', 'fsl'):
            raise NiAnalysisError(
                "Unrecognised value for 'bias_method' option '{}'. It can "
                "be one of 'ants' or 'fsl'.".format(bias_method))
        pipeline = self.create_pipeline(
            name='bias_correct',
            inputs=[DatasetSpec('dwi_preproc', nifti_gz_format),
                    DatasetSpec('brain_mask', nifti_gz_format),
                    DatasetSpec('grad_dirs', fsl_bvecs_format),
                    DatasetSpec('bvalues', fsl_bvals_format)],
            outputs=[DatasetSpec('bias_correct', nifti_gz_format)],
            description="Corrects for B1 field inhomogeneity",
            default_options={'method': bias_method_default},
            version=1,
            requirements=[mrtrix3_req,
                          (ants2_req if bias_method == 'ants' else fsl5_req)],
            citations=[fast_cite,
                       (n4_cite if bias_method == 'ants' else fsl_cite)],
            approx_runtime=1,
            options=options)
        # Create bias correct node
        bias_correct = pipeline.create_node(DWIBiasCorrect(), name="bias_correct")
        bias_correct.inputs.method = bias_method
        # Gradient merge node
        fsl_grads = pipeline.create_node(MergeTuple(2), name="fsl_grads")
        # Connect nodes
        pipeline.connect(fsl_grads, 'out', bias_correct, 'fslgrad')
        # Connect to inputs
        pipeline.connect_input('grad_dirs', fsl_grads, 'in1')
        pipeline.connect_input('bvalues', fsl_grads, 'in2')
        pipeline.connect_input('dwi_preproc', bias_correct, 'in_file')
        pipeline.connect_input('brain_mask', bias_correct, 'mask')
        # Connect to outputs
        pipeline.connect_output('bias_correct', bias_correct, 'out_file')
        # Check inputs/output are connected
        pipeline.assert_connected()
        return pipeline

    def tensor_pipeline(self, **options):  # @UnusedVariable
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
            description=("Estimates the apparrent diffusion tensor in each "
                         "voxel"),
            default_options={},
            version=1,
            citations=[],
            requirements=[mrtrix3_req],
            approx_runtime=1,
            options=options)
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
        pipeline.assert_connected()
        return pipeline

    def fa_pipeline(self, **options):  # @UnusedVariable
        """
        Fits the apparrent diffusion tensor (DT) to each voxel of the image
        """
        pipeline = self.create_pipeline(
            name='fa',
            inputs=[DatasetSpec('tensor', nifti_gz_format),
                    DatasetSpec('brain_mask', nifti_gz_format)],
            outputs=[DatasetSpec('fa', nifti_gz_format),
                     DatasetSpec('adc', nifti_gz_format)],
            description=("Calculates the FA and ADC from a tensor image"),
            default_options={},
            version=1,
            citations=[],
            requirements=[mrtrix3_req],
            approx_runtime=1,
            options=options)
        # Create tensor fit node
        metrics = pipeline.create_node(TensorMetrics(), name='metrics')
        metrics.inputs.out_fa = 'fa.nii.gz'
        metrics.inputs.out_adc = 'adc.nii.gz'
        # Connect to inputs
        pipeline.connect_input('tensor', metrics, 'in_file')
        pipeline.connect_input('brain_mask', metrics, 'in_mask')
        # Connect to outputs
        pipeline.connect_output('fa', metrics, 'out_fa')
        pipeline.connect_output('adc', metrics, 'out_adc')
        # Check inputs/output are connected
        pipeline.assert_connected()
        return pipeline

    def fod_pipeline(self, **options):  # @UnusedVariable
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
                    DatasetSpec('brain_mask', nifti_gz_format)],
            outputs=[DatasetSpec('fod', nifti_gz_format)],
            description=("Estimates the fibre orientation distribution in each"
                         " voxel"),
            default_options={},
            version=1,
            citations=[mrtrix_cite],
            requirements=[mrtrix3_req],
            approx_runtime=1,
            options=options)
        # Create fod fit node
        dwi2fod = pipeline.create_node(EstimateFOD(), name='dwi2fod')
        response = pipeline.create_node(ResponseSD(), name='response')
        # Gradient merge node
        fsl_grads = pipeline.create_node(MergeTuple(2), name="fsl_grads")
        # Connect nodes
        pipeline.connect(fsl_grads, 'out', response, 'grad_fsl')
        pipeline.connect(fsl_grads, 'out', dwi2fod, 'grad_fsl')
        pipeline.connect(response, 'out_file', dwi2fod, 'response')
        # Connect to inputs
        pipeline.connect_input('grad_dirs', fsl_grads, 'in1')
        pipeline.connect_input('bvalues', fsl_grads, 'in2')
        pipeline.connect_input('bias_correct', dwi2fod, 'in_file')
        pipeline.connect_input('bias_correct', response, 'in_file')
        pipeline.connect_input('brain_mask', response, 'in_mask')
        # Connect to outputs
        pipeline.connect_output('fod', dwi2fod, 'out_file')
        # Check inputs/output are connected
        pipeline.assert_connected()
        return pipeline

    def tbss_pipeline(self, **options):  # @UnusedVariable
        pipeline = self.create_pipeline(
            name='tbss',
            inputs=[DatasetSpec('fa', nifti_gz_format)],
            outputs=[DatasetSpec('tbss_mean_fa', nifti_gz_format),
                     DatasetSpec('tbss_proj_fa', nifti_gz_format),
                     DatasetSpec('tbss_skeleton', nifti_gz_format),
                     DatasetSpec('tbss_skeleton_mask', nifti_gz_format)],
            default_options={'tbss_skel_thresh': 0.2},
            version=1,
            citations=[tbss_cite, fsl_cite],
            requirements=[fsl5_req],
            approx_runtime=1,
            options=options)
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
        pipeline.assert_connected()
        return pipeline

    def extract_b0_pipeline(self, **options):  # @UnusedVariable
        """
        Extracts the b0 images from a DWI study and takes their mean
        """
        pipeline = self.create_pipeline(
            name='extract_b0',
            inputs=[DatasetSpec('bias_correct', nifti_gz_format),
                    DatasetSpec('grad_dirs', fsl_bvecs_format),
                    DatasetSpec('bvalues', fsl_bvals_format)],
            outputs=[DatasetSpec('primary', nifti_gz_format)],
            description="Extract b0 image from a DWI study",
            default_options={},
            version=1,
            requirements=[mrtrix3_req],
            citations=[mrtrix_cite],
            approx_runtime=0.5,
            options=options)
        # Gradient merge node
        fsl_grads = pipeline.create_node(MergeTuple(2), name="fsl_grads")
        # Extraction node
        extract_b0s = pipeline.create_node(ExtractDWIorB0(), name='extract_b0s')
        extract_b0s.inputs.bzero = True
        extract_b0s.inputs.quiet = True
        # FIXME: Need a registration step before the mean
        # Mean calculation node
        mean = pipeline.create_node(MRMath(), name="mean")
        mean.inputs.axis = 3
        mean.inputs.operation = 'mean'
        mean.inputs.quiet = True
        # Convert to Nifti
        mrconvert = pipeline.create_node(MRConvert(), name="output_conversion")
        mrconvert.inputs.out_ext = '.nii.gz'
        mrconvert.inputs.quiet = True
        # Connect inputs
        pipeline.connect_input('bias_correct', extract_b0s, 'in_file')
        pipeline.connect_input('grad_dirs', fsl_grads, 'in1')
        pipeline.connect_input('bvalues', fsl_grads, 'in2')
        # Connect between nodes
        pipeline.connect(extract_b0s, 'out_file', mean, 'in_files')
        pipeline.connect(fsl_grads, 'out', extract_b0s, 'fslgrad')
        pipeline.connect(mean, 'out_file', mrconvert, 'in_file')
        # Connect outputs
        pipeline.connect_output('primary', mrconvert, 'out_file')
        pipeline.assert_connected()
        # Check inputs/outputs are connected
        return pipeline

    def track_gen_pipeline(self, **options):
        pipeline = self.create_pipeline(
            name='extract_b0',
            inputs=[DatasetSpec('bias_correct', nifti_gz_format),
                    DatasetSpec('grad_dirs', fsl_bvecs_format),
                    DatasetSpec('bvalues', fsl_bvals_format)],
            outputs=[DatasetSpec('primary', nifti_gz_format)],
            description="Extract b0 image from a DWI study",
            default_options={},
            version=1,
            requirements=[mrtrix3_req],
            citations=[mrtrix_cite], approx_runtime=0.5, options=options)
        pipeline.assert_connected()
        return pipeline

    # The list of study dataset_specs that are either primary from the scanner
    # (i.e. without a specified pipeline) or generated by processing pipelines
    _dataset_specs = set_dataset_specs(
        DatasetSpec('dwi_scan', mrtrix_format),
        DatasetSpec('forward_rpe', mrtrix_format),
        DatasetSpec('reverse_rpe', mrtrix_format),
        DatasetSpec('primary', nifti_gz_format, extract_b0_pipeline,
                    description="b0 image"),
        DatasetSpec('tensor', nifti_gz_format, tensor_pipeline),
        DatasetSpec('fa', nifti_gz_format, tensor_pipeline),
        DatasetSpec('adc', nifti_gz_format, tensor_pipeline),
        DatasetSpec('fod', mrtrix_format, tensor_pipeline),
        DatasetSpec('dwi_preproc', nifti_gz_format, preprocess_pipeline),
        DatasetSpec('bias_correct', nifti_gz_format, bias_correct_pipeline),
        DatasetSpec('grad_dirs', fsl_bvecs_format, preprocess_pipeline),
        DatasetSpec('bvalues', fsl_bvals_format, preprocess_pipeline),
        DatasetSpec('tbss_mean_fa', nifti_gz_format, tbss_pipeline,
                    multiplicity='per_project'),
        DatasetSpec('tbss_proj_fa', nifti_gz_format, tbss_pipeline,
                    multiplicity='per_project'),
        DatasetSpec('tbss_skeleton', nifti_gz_format, tbss_pipeline,
                    multiplicity='per_project'),
        DatasetSpec('tbss_skeleton_mask', nifti_gz_format, tbss_pipeline,
                    multiplicity='per_project'),
        DatasetSpec('masked', nifti_gz_format, brain_mask_pipeline),
        DatasetSpec('brain_mask', nifti_gz_format, brain_mask_pipeline),
        inherit_from=T2Study.generated_dataset_specs())


# class TractographyInputSpec(MRTrix3BaseInputSpec):
#     sph_trait = traits.Tuple(traits.Float, traits.Float, traits.Float,
#                              traits.Float, argstr='%f,%f,%f,%f')
#
#     in_file = File(exists=True, argstr='%s', mandatory=True, position=-2,
#                    desc='input file to be processed')
#
#     out_file = File('tracked.tck', argstr='%s', mandatory=True, position=-1,
#                     usedefault=True, desc='output file containing tracks')
#
#     algorithm = traits.Enum(
#         'iFOD2', 'FACT', 'iFOD1', 'Nulldist', 'SD_Stream', 'Tensor_Det',
#         'Tensor_Prob', usedefault=True, argstr='-algorithm %s',
#         desc='tractography algorithm to be used')
#
#     # ROIs processing options
#     roi_incl = traits.Either(
#         File(exists=True), sph_trait, argstr='-include %s',
#         desc=('specify an inclusion region of interest, streamlines must'
#               ' traverse ALL inclusion regions to be accepted'))
#     roi_excl = traits.Either(
#         File(exists=True), sph_trait, argstr='-exclude %s',
#         desc=('specify an exclusion region of interest, streamlines that'
#               ' enter ANY exclude region will be discarded'))
#     roi_mask = traits.Either(
#         File(exists=True), sph_trait, argstr='-mask %s',
#         desc=('specify a masking region of interest. If defined,'
#               'streamlines exiting the mask will be truncated'))
#
#     # Streamlines tractography options
#     step_size = traits.Float(
#         argstr='-step %f',
#         desc=('set the step size of the algorithm in mm (default is 0.1'
#               ' x voxelsize; for iFOD2: 0.5 x voxelsize)'))
#     angle = traits.Float(
#         argstr='-angle %f',
#         desc=('set the maximum angle between successive steps (default '
#               'is 90deg x stepsize / voxelsize)'))
#     n_tracks = traits.Int(
#         argstr='-number %d',
#         desc=('set the desired number of tracks. The program will continue'
#               ' to generate tracks until this number of tracks have been '
#               'selected and written to the output file'))
#     max_tracks = traits.Int(
#         argstr='-maxnum %d',
#         desc=('set the maximum number of tracks to generate. The program '
#               'will not generate more tracks than this number, even if '
#               'the desired number of tracks hasn\'t yet been reached '
#               '(default is 100 x number)'))
#     max_length = traits.Float(
#         argstr='-maxlength %f',
#         desc=('set the maximum length of any track in mm (default is '
#               '100 x voxelsize)'))
#     min_length = traits.Float(
#         argstr='-minlength %f',
#         desc=('set the minimum length of any track in mm (default is '
#               '5 x voxelsize)'))
#     cutoff = traits.Float(
#         argstr='-cutoff %f',
#         desc=('set the FA or FOD amplitude cutoff for terminating '
#               'tracks (default is 0.1)'))
#     cutoff_init = traits.Float(
#         argstr='-initcutoff %f',
#         desc=('set the minimum FA or FOD amplitude for initiating '
#               'tracks (default is the same as the normal cutoff)'))
#     n_trials = traits.Int(
#         argstr='-trials %d',
#         desc=('set the maximum number of sampling trials at each point'
#               ' (only used for probabilistic tracking)'))
#     unidirectional = traits.Bool(
#         argstr='-unidirectional',
#         desc=('track from the seed point in one direction only '
#               '(default is to track in both directions)'))
#     init_dir = traits.Tuple(
#         traits.Float, traits.Float, traits.Float,
#         argstr='-initdirection %f,%f,%f',
#         desc=('specify an initial direction for the tracking (this '
#               'should be supplied as a vector of 3 comma-separated values'))
#     noprecompt = traits.Bool(
#         argstr='-noprecomputed',
#         desc=('do NOT pre-compute legendre polynomial values. Warning: this '
#               'will slow down the algorithm by a factor of approximately 4'))
#     power = traits.Int(
#         argstr='-power %d',
#         desc=('raise the FOD to the power specified (default is 1/nsamples)'))
#     n_samples = traits.Int(
#         4, argstr='-samples %d',
#         desc=('set the number of FOD samples to take per step for the 2nd '
#               'order (iFOD2) method'))
#     use_rk4 = traits.Bool(
#         argstr='-rk4',
#         desc=('use 4th-order Runge-Kutta integration (slower, but eliminates'
#               ' curvature overshoot in 1st-order deterministic methods)'))
#     stop = traits.Bool(
#         argstr='-stop',
#         desc=('stop propagating a streamline once it has traversed all '
#               'include regions'))
#     downsample = traits.Float(
#         argstr='-downsample %f',
#         desc='downsample the generated streamlines to reduce output file size')
#
#     # Anatomically-Constrained Tractography options
#     act_file = File(
#         exists=True, argstr='-act %s',
#         desc=('use the Anatomically-Constrained Tractography framework during'
#               ' tracking; provided image must be in the 5TT '
#               '(five - tissue - type) format'))
#     backtrack = traits.Bool(argstr='-backtrack',
#                             desc='allow tracks to be truncated')
#
#     crop_at_gmwmi = traits.Bool(
#         argstr='-crop_at_gmwmi',
#         desc=('crop streamline endpoints more '
#               'precisely as they cross the GM-WM interface'))
#
#     # Tractography seeding options
#     seed_sphere = traits.Tuple(
#         traits.Float, traits.Float, traits.Float, traits.Float,
#         argstr='-seed_sphere %f,%f,%f,%f', desc='spherical seed')
#     seed_image = File(exists=True, argstr='-seed_image %s',
#                       desc='seed streamlines entirely at random within mask')
#     seed_rnd_voxel = traits.Tuple(
#         File(exists=True), traits.Int(),
#         argstr='-seed_random_per_voxel %s %d',
#         xor=['seed_image', 'seed_grid_voxel'],
#         desc=('seed a fixed number of streamlines per voxel in a mask '
#               'image; random placement of seeds in each voxel'))
#     seed_grid_voxel = traits.Tuple(
#         File(exists=True), traits.Int(),
#         argstr='-seed_grid_per_voxel %s %d',
#         xor=['seed_image', 'seed_rnd_voxel'],
#         desc=('seed a fixed number of streamlines per voxel in a mask '
#               'image; place seeds on a 3D mesh grid (grid_size argument '
#               'is per axis; so a grid_size of 3 results in 27 seeds per'
#               ' voxel)'))
#     seed_rejection = File(
#         exists=True, argstr='-seed_rejection %s',
#         desc=('seed from an image using rejection sampling (higher '
#               'values = more probable to seed from'))
#     seed_gmwmi = File(
#         exists=True, argstr='-seed_gmwmi %s', requires=['act_file'],
#         desc=('seed from the grey matter - white matter interface (only '
#               'valid if using ACT framework)'))
#     seed_dynamic = File(
#         exists=True, argstr='-seed_dynamic %s',
#         desc=('determine seed points dynamically using the SIFT model '
#               '(must not provide any other seeding mechanism). Note that'
#               ' while this seeding mechanism improves the distribution of'
#               ' reconstructed streamlines density, it should NOT be used '
#               'as a substitute for the SIFT method itself.'))
#     max_seed_attempts = traits.Int(
#         argstr='-max_seed_attempts %d',
#         desc=('set the maximum number of times that the tracking '
#               'algorithm should attempt to find an appropriate tracking'
#               ' direction from a given seed point'))
#     out_seeds = File(
#         'out_seeds.nii.gz', argstr='-output_seeds %s',
#         desc=('output the seed location of all successful streamlines to'
#               ' a file'))


class NODDIStudy(DiffusionStudy):

    def concatenate_pipeline(self, **options):  # @UnusedVariable
        """
        Concatenates two dMRI datasets (with different b-values) along the
        DW encoding (4th) axis
        """
        pipeline = self.create_pipeline(
            name='concatenation',
            inputs=[DatasetSpec('low_b_dw_scan', mrtrix_format),
                    DatasetSpec('high_b_dw_scan', mrtrix_format)],
            outputs=[DatasetSpec('dwi_scan', mrtrix_format)],
            description=(
                "Concatenate low and high b-value dMRI datasets for NODDI "
                "processing"),
            default_options={},
            version=1,
            requirements=[mrtrix3_req],
            citations=[mrtrix_cite], approx_runtime=1,
            options=options)
        # Create concatenation node
        mrcat = pipeline.create_node(MRCat(), name='mrcat')
        mrcat.inputs.quiet = True
        # Output conversion to nifti_gz
        mrconvert = pipeline.create_node(MRConvert(), name="output_conversion")
        mrconvert.inputs.out_ext = '.nii.gz'
        mrconvert.inputs.quiet = True
        # Connect nodes
        pipeline.connect(mrcat, 'out_file', mrconvert, 'in_file')
        # Connect inputs
        pipeline.connect_input('low_b_dw_scan', mrcat, 'first_scan')
        pipeline.connect_input('high_b_dw_scan', mrcat, 'second_scan')
        # Connect outputs
        pipeline.connect_output('dwi_scan', mrconvert, 'out_file')
        # Check inputs/outputs are connected
        pipeline.assert_connected()
        return pipeline

    def noddi_fitting_pipeline(self, nthreads=4, **options):  # @UnusedVariable
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
        inputs = [DatasetSpec('bias_correct', nifti_gz_format),
                  DatasetSpec('grad_dirs', fsl_bvecs_format),
                  DatasetSpec('bvalues', fsl_bvals_format)]
        if options.get('single_slice', False):
            inputs.append(DatasetSpec('eroded_mask', nifti_gz_format))
        else:
            inputs.append(DatasetSpec('brain_mask', nifti_gz_format))
        pipeline = self.create_pipeline(
            name='noddi_fitting',
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
            description=(
                "Creates a ROI in which the NODDI processing will be "
                "performed"),
            default_options={'noddi_model': 'WatsonSHStickTortIsoV_B0',
                             'single_slice': False},
            requirements=[Requirement('matlab', min_version=(2016, 'a')),
                          Requirement('noddi', min_version=(0, 9)),
                          Requirement('niftimatlib', (1, 2))],
            citations=[noddi_cite], approx_runtime=60,
            options=options)
        # Create node to unzip the nifti files
        unzip_bias_correct = pipeline.create_node(MRConvert(), name="unzip_bias_correct")
        unzip_bias_correct.inputs.out_ext = 'nii'
        unzip_bias_correct.inputs.quiet = True
        unzip_mask = pipeline.create_node(MRConvert(), name="unzip_mask")
        unzip_mask.inputs.out_ext = 'nii'
        unzip_mask.inputs.quiet = True
        # Create create-roi node
        create_roi = pipeline.create_node(CreateROI(), name='create_roi')
        pipeline.connect(unzip_bias_correct, 'out_file', create_roi, 'in_file')
        pipeline.connect(unzip_mask, 'out_file', create_roi, 'brain_mask')
        # Create batch-fitting node
        batch_fit = pipeline.create_node(BatchNODDIFitting(), name="batch_fit")
        batch_fit.inputs.model = pipeline.option('noddi_model')
        batch_fit.inputs.nthreads = nthreads
        pipeline.connect(create_roi, 'out_file', batch_fit, 'roi_file')
        # Create output node
        save_params = pipeline.create_node(SaveParamsAsNIfTI(), name="save_params")
        save_params.inputs.output_prefix = 'params'
        pipeline.connect(batch_fit, 'out_file', save_params, 'params_file')
        pipeline.connect(create_roi, 'out_file', save_params, 'roi_file')
        pipeline.connect(unzip_mask, 'out_file', save_params,
                         'brain_mask_file')
        # Connect inputs
        pipeline.connect_input('bias_correct', unzip_bias_correct, 'in_file')
        if pipeline.option('single_slice') is None:
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
        pipeline.assert_connected()
        return pipeline

    _dataset_specs = set_dataset_specs(
        DatasetSpec('low_b_dw_scan', mrtrix_format),
        DatasetSpec('high_b_dw_scan', mrtrix_format),
        DatasetSpec('forward_rpe', mrtrix_format),
        DatasetSpec('reverse_rpe', mrtrix_format),
        DatasetSpec('dwi_scan', mrtrix_format, concatenate_pipeline),
        DatasetSpec('ficvf', nifti_format, noddi_fitting_pipeline),
        DatasetSpec('odi', nifti_format, noddi_fitting_pipeline),
        DatasetSpec('fiso', nifti_format, noddi_fitting_pipeline),
        DatasetSpec('fibredirs_xvec', nifti_format, noddi_fitting_pipeline),
        DatasetSpec('fibredirs_yvec', nifti_format, noddi_fitting_pipeline),
        DatasetSpec('fibredirs_zvec', nifti_format, noddi_fitting_pipeline),
        DatasetSpec('fmin', nifti_format, noddi_fitting_pipeline),
        DatasetSpec('kappa', nifti_format, noddi_fitting_pipeline),
        DatasetSpec('error_code', nifti_format, noddi_fitting_pipeline),
        inherit_from=DiffusionStudy.generated_dataset_specs())
