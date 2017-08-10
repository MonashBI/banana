import os.path
import errno
import math
from nipype.interfaces.base import (
    CommandLineInputSpec, CommandLine, File, Directory, TraitedSpec, isdefined,
    traits, InputMultiPath)
from nipype.interfaces.mrtrix3.reconst import (
    MRTrix3Base, MRTrix3BaseInputSpec)
from nipype.interfaces.mrtrix3.preprocess import (
    ResponseSD as NipypeResponseSD,
    ResponseSDInputSpec as NipypeResponseSDInputSpec)
from nianalysis.utils import split_extension


# TODO: Write MRtrixBaseInputSpec with all the generic options included

# =============================================================================
# Extract MR gradients
# =============================================================================

class ResponseSDInputSpec(NipypeResponseSDInputSpec):

    algorithm = traits.Str(mandatory=True, argstr='%s', position=0,
                           desc="The algorithm used to estimate the response")


class ResponseSD(NipypeResponseSD):

    input_spec = ResponseSDInputSpec


class ExtractFSLGradientsInputSpec(CommandLineInputSpec):
    in_file = File(exists=True, argstr='%s', mandatory=True, position=0,
                   desc="Diffusion weighted images with graident info")
    bvecs_file = File(genfile=True, argstr='-export_grad_fsl %s', position=1,
                      desc=("Extracted gradient encoding directions in FSL "
                            "format"))
    bvals_file = File(genfile=True, argstr='%s', position=2,
                      desc=("Extracted graident encoding b-values in FSL "
                            "format"))


class ExtractFSLGradientsOutputSpec(TraitedSpec):
    bvecs_file = File(exists=True,
                      desc='Extracted encoding gradient directions')
    bvals_file = File(exists=True,
                      desc='Extracted encoding gradient b-values')


class ExtractFSLGradients(CommandLine):
    """
    Extracts the gradient information in MRtrix format from a DWI image
    """
    _cmd = 'mrinfo'
    input_spec = ExtractFSLGradientsInputSpec
    output_spec = ExtractFSLGradientsOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['bvecs_file'] = self._gen_grad_filename('bvec')
        outputs['bvals_file'] = self._gen_grad_filename('bval')
        return outputs

    def _gen_filename(self, name):
        if name == 'bvecs_file':
            fname = self._gen_grad_filename('bvec')
        elif name == 'bvals_file':
            fname = self._gen_grad_filename('bval')
        else:
            assert False
        return fname

    def _gen_grad_filename(self, comp):
        filename = getattr(self.inputs, comp + 's_file')
        if not isdefined(filename):
            base, _ = split_extension(os.path.basename(self.inputs.in_file))
            filename = os.path.join(
                os.getcwd(), "{base}_{comp}s.{comp}".format(base=base,
                                                            comp=comp))
        return filename


# =============================================================================
# MR math
# =============================================================================

class MRMathInputSpec(CommandLineInputSpec):

    in_files = InputMultiPath(
        File(exists=True), argstr='%s', mandatory=True,
        position=3, desc="Diffusion weighted images with graident info")

    out_file = File(genfile=True, argstr='%s', position=-1,
                    desc="Extracted DW or b-zero images")

    operation = traits.Str(mandatory=True, argstr='%s', position=-2,  # @UndefinedVariable @IgnorePep8
                           desc=("Operation to apply to the files"))

    axis = traits.Int(argstr="-axis %s", position=0,  # @UndefinedVariable @IgnorePep8
                      desc=("The axis over which to apply the operator"))

    quiet = traits.Bool(
        mandatory=False, argstr="-quiet",
        description="Don't display output during operation")


class MRMathOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='The resultant image')


class MRMath(CommandLine):
    """
    Extracts the gradient information in MRtrix format from a DWI image
    """
    _cmd = 'mrmath'
    input_spec = MRMathInputSpec
    output_spec = MRMathOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = self._gen_outfilename()
        return outputs

    def _gen_filename(self, name):
        if name == 'out_file':
            fname = self._gen_outfilename()
        else:
            assert False
        return fname

    def _gen_outfilename(self):
        if isdefined(self.inputs.out_file):
            filename = self.inputs.out_file
        else:
            base, ext = split_extension(
                os.path.basename(self.inputs.in_files[0]))
            filename = os.path.join(
                os.getcwd(),
                "{}_{}{}".format(base, self.inputs.operation, ext))
        return filename


# =============================================================================
# MR math
# =============================================================================

class MRCalcInputSpec(CommandLineInputSpec):

    operands = traits.List(
        traits.Any(), argstr='%s',
        mandatory=True, position=-3,
        desc="Diffusion weighted images with graident info")

    out_file = File(genfile=True, argstr='%s', position=-1,
                    desc="Extracted DW or b-zero images")

    operation = traits.Enum(
        'abs', 'neg', 'sqrt', 'exp', 'log', 'log10', 'cos', 'sin', 'tan',
        'cosh', 'sinh', 'tanh', 'acos', 'asin', 'atan', 'acosh', 'asinh',
        'atanh', 'round', 'ceil', 'floor', 'isnan', 'isinf', 'finite', 'real',
        'imag', 'phase', 'conj', 'add', 'subtract', 'multiply', 'divide',
        'pow', 'min', 'max', 'lt', 'gt', 'le', 'ge', 'eq', 'neq', 'complex',
        'if', mandatory=True, argstr='-%s',
        position=-2, desc=("Operation to apply to the files"))

    quiet = traits.Bool(
        mandatory=False, argstr="-quiet",
        description="Don't display output during operation")


class MRCalcOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='The resultant image')


class MRCalc(CommandLine):
    """
    Extracts the gradient information in MRtrix format from a DWI image
    """
    _cmd = 'mrcalc'
    input_spec = MRCalcInputSpec
    output_spec = MRCalcOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = self._gen_outfilename()
        return outputs

    def _gen_filename(self, name):
        if name == 'out_file':
            fname = self._gen_outfilename()
        else:
            assert False
        return fname

    def _gen_outfilename(self):
        if isdefined(self.inputs.out_file):
            filename = self.inputs.out_file
        else:
            _, ext = split_extension(
                os.path.basename(self.inputs.operands[0]))
            filename = os.getcwd()
            for op in self.inputs.operands:
                try:
                    op_str = split_extension(os.path.basename(op))[0]
                except:
                    op_str = str(float(op))
                filename += '_' + op_str
            filename += '_' + self.inputs.operation + ext
        return filename


class EstimateFODInputSpec(MRTrix3BaseInputSpec):

    algorithm = traits.Enum('csd', 'msmt_csd', mandatory=True, position=0,
                            desc="Algorithm used for CSD estimation")

    in_file = File(exists=True, argstr='%s', mandatory=True, position=-3,
                   desc='input diffusion weighted images')
    response = File(
        exists=True, argstr='%s', mandatory=True, position=-2,
        desc=('a text file containing the diffusion-weighted signal response '
              'function coefficients for a single fibre population'))
    out_file = File(
        'fods.mif', argstr='%s', mandatory=True, position=-1,
        usedefault=True, desc=('the output spherical harmonics coefficients'
                               ' image'))

    # DW Shell selection options
    shell = traits.List(traits.Float, sep=',', argstr='-shell %s',
                        desc='specify one or more dw gradient shells')

    # Spherical deconvolution options
    max_sh = traits.Int(8, argstr='-lmax %d',
                        desc='maximum harmonic degree of response function')
    in_mask = File(exists=True, argstr='-mask %s',
                   desc='provide initial mask image')
    in_dirs = File(
        exists=True, argstr='-directions %s',
        desc=('specify the directions over which to apply the non-negativity '
              'constraint (by default, the built-in 300 direction set is '
              'used). These should be supplied as a text file containing the '
              '[ az el ] pairs for the directions.'))
    sh_filter = File(
        exists=True, argstr='-filter %s',
        desc=('the linear frequency filtering parameters used for the initial '
              'linear spherical deconvolution step (default = [ 1 1 1 0 0 ]). '
              'These should be supplied as a text file containing the '
              'filtering coefficients for each even harmonic order.'))

    neg_lambda = traits.Float(
        1.0, argstr='-neg_lambda %f',
        desc=('the regularisation parameter lambda that controls the strength'
              ' of the non-negativity constraint'))
    thres = traits.Float(
        0.0, argstr='-threshold %f',
        desc=('the threshold below which the amplitude of the FOD is assumed '
              'to be zero, expressed as an absolute amplitude'))

    n_iter = traits.Int(
        50, argstr='-niter %d', desc=('the maximum number of iterations '
                                      'to perform for each voxel'))


class EstimateFODOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='the output response file')


class EstimateFOD(MRTrix3Base):

    """
    Convert diffusion-weighted images to tensor images

    Note that this program makes use of implied symmetries in the diffusion
    profile. First, the fact the signal attenuation profile is real implies
    that it has conjugate symmetry, i.e. Y(l,-m) = Y(l,m)* (where * denotes
    the complex conjugate). Second, the diffusion profile should be
    antipodally symmetric (i.e. S(x) = S(-x)), implying that all odd l
    components should be zero. Therefore, this program only computes the even
    elements.

    Note that the spherical harmonics equations used here differ slightly from
    those conventionally used, in that the (-1)^m factor has been omitted.
    This should be taken into account in all subsequent calculations.
    The spherical harmonic coefficients are stored as follows. First, since
    the signal attenuation profile is real, it has conjugate symmetry, i.e.
    Y(l,-m) = Y(l,m)* (where * denotes the complex conjugate). Second, the
    diffusion profile should be antipodally symmetric (i.e. S(x) = S(-x)),
    implying that all odd l components should be zero. Therefore, only the
    even elements are computed.

    Note that the spherical harmonics equations used here differ slightly from
    those conventionally used, in that the (-1)^m factor has been omitted.
    This should be taken into account in all subsequent calculations.
    Each volume in the output image corresponds to a different spherical
    harmonic component. Each volume will correspond to the following:

    volume 0: l = 0, m = 0
    volume 1: l = 2, m = -2 (imaginary part of m=2 SH)
    volume 2: l = 2, m = -1 (imaginary part of m=1 SH)
    volume 3: l = 2, m = 0
    volume 4: l = 2, m = 1 (real part of m=1 SH)
    volume 5: l = 2, m = 2 (real part of m=2 SH)
    etc...



    Example
    -------

    >>> import nipype.interfaces.mrtrix3 as mrt
    >>> fod = mrt.EstimateFOD()
    >>> fod.inputs.in_file = 'dwi.mif'
    >>> fod.inputs.response = 'response.txt'
    >>> fod.inputs.in_mask = 'mask.nii.gz'
    >>> fod.inputs.grad_fsl = ('bvecs', 'bvals')
    >>> fod.cmdline                               # doctest: +ELLIPSIS +ALLOW_UNICODE
    'dwi2fod -fslgrad bvecs bvals -mask mask.nii.gz dwi.mif response.txt\
 fods.mif'
    >>> fod.run()                                 # doctest: +SKIP
    """

    _cmd = 'dwi2fod'
    input_spec = EstimateFODInputSpec
    output_spec = EstimateFODOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = os.path.abspath(self.inputs.out_file)
        return outputs


# =============================================================================
# MR Crop
# =============================================================================

class MRCropInputSpec(CommandLineInputSpec):
    in_file = File(exists=True, argstr='%s', mandatory=True, position=-2,
                   desc="Diffusion weighted images with graident info")

    out_file = File(genfile=True, argstr='%s', position=-1,
                    desc="Extracted DW or b-zero images")

    axis = traits.Tuple(
        traits.Int(desc="index"),  # @UndefinedVariable
        traits.Int(desc="start"),  # @UndefinedVariable
        traits.Int(desc='end'),  # @UndefinedVariable
        mandatory=False, argstr="-axis %s %s %s", # @UndefinedVariable @IgnorePep8
        desc=("crop the input image in the provided axis"))

    mask = File(mandatory=False, exists=True, argstr="-mask %s",
                desc=("Crop the input image according to the spatial extent of"
                      " a mask image"))

    quiet = traits.Bool(
        mandatory=False, argstr="-quiet",
        description="Don't display output during operation")


class MRCropOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='The resultant image')


class MRCrop(CommandLine):
    """
    Extracts the gradient information in MRtrix format from a DWI image
    """
    _cmd = 'mrcrop'
    input_spec = MRCropInputSpec
    output_spec = MRCropOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = self._gen_outfilename()
        return outputs

    def _gen_filename(self, name):
        if name == 'out_file':
            fname = self._gen_outfilename()
        else:
            assert False
        return fname

    def _gen_outfilename(self):
        if isdefined(self.inputs.out_file):
            filename = self.inputs.out_file
        else:
            base, ext = split_extension(os.path.basename(self.inputs.in_file))
            filename = os.path.join(os.getcwd(),
                                    "{}_crop{}".format(base, ext))
        return filename


# =============================================================================
# MR Pad
# =============================================================================

class MRPadInputSpec(CommandLineInputSpec):
    in_file = File(exists=True, argstr='%s', mandatory=True, position=-2,
                   desc="Diffusion weighted images with graident info")

    out_file = File(genfile=True, argstr='%s', position=-1,
                    desc="Extracted DW or b-zero images")

    axis = traits.Tuple(
        traits.Int(desc="index"),  # @UndefinedVariable
        traits.Int(desc="lower"),  # @UndefinedVariable
        traits.Int(desc='upper'),  # @UndefinedVariable
        mandatory=False, argstr="-axis %s %s %s", # @UndefinedVariable @IgnorePep8
        desc=("Pad the input image along the provided axis (defined by index)."
              "Lower and upper define the number of voxels to add to the lower"
              " and upper bounds of the axis"))

    uniform = File(mandatory=False, exists=True, argstr="-uniform %s",
                   desc=("Pad the input image by a uniform number of voxels on"
                         " all sides (in 3D)"))

    quiet = traits.Bool(
        mandatory=False, argstr="-quiet",
        description="Don't display output during operation")


class MRPadOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='The resultant image')


class MRPad(CommandLine):
    """
    Extracts the gradient information in MRtrix format from a DWI image
    """
    _cmd = 'mrpad'
    input_spec = MRPadInputSpec
    output_spec = MRPadOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = self._gen_outfilename()
        return outputs

    def _gen_filename(self, name):
        if name == 'out_file':
            fname = self._gen_outfilename()
        else:
            assert False
        return fname

    def _gen_outfilename(self):
        if isdefined(self.inputs.out_file):
            filename = self.inputs.out_file
        else:
            base, ext = split_extension(os.path.basename(self.inputs.in_file))
            filename = os.path.join(os.getcwd(),
                                    "{}_pad{}".format(base, ext))
        return filename


# =============================================================================
# Extract b0 or DW images
# =============================================================================

class ExtractDWIorB0InputSpec(CommandLineInputSpec):
    in_file = File(exists=True, argstr='%s', mandatory=True, position=0,
                   desc="Diffusion weighted images with graident info")

    out_file = File(genfile=True, argstr='%s', position=-1,
                    desc="Extracted DW or b-zero images")

    bzero = traits.Bool(argstr='-bzero', position=1,  # @UndefinedVariable
                        desc="Extract b-zero images instead of DDW images")

    quiet = traits.Bool(
        mandatory=False, argstr="-quiet",
        description="Don't display output during operation")

    grad = traits.Str(
        mandatory=False, argstr='-grad %s',
        desc=("specify the diffusion-weighted gradient scheme used in the  "
              "acquisition. The program will normally attempt to use the  "
              "encoding stored in the image header. This should be supplied  "
              "as a 4xN text file with each line is in the format [ X Y Z b ],"
              " where [ X Y Z ] describe the direction of the applied  "
              "gradient, and b gives the b-value in units of s/mm^2."))

    fslgrad = traits.Tuple(
        File(exists=True, desc="gradient directions file (bvec)"),  # @UndefinedVariable @IgnorePep8
        File(exists=True, desc="b-values (bval)"),  # @UndefinedVariable @IgnorePep8
        argstr='-fslgrad %s %s', mandatory=False,
        desc=("specify the diffusion-weighted gradient scheme used in the "
              "acquisition in FSL bvecs/bvals format."))


class ExtractDWIorB0OutputSpec(TraitedSpec):

    out_file = File(exists=True, desc='Extracted DW or b-zero images')


class ExtractDWIorB0(CommandLine):
    """
    Extracts the gradient information in MRtrix format from a DWI image
    """
    _cmd = 'dwiextract'
    input_spec = ExtractDWIorB0InputSpec
    output_spec = ExtractDWIorB0OutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = self._gen_outfilename()
        return outputs

    def _gen_filename(self, name):
        if name == 'out_file':
            fname = self._gen_outfilename()
        else:
            assert False
        return fname

    def _gen_outfilename(self):
        if isdefined(self.inputs.out_file):
            filename = self.inputs.out_file
        else:
            base, ext = split_extension(os.path.basename(self.inputs.in_file))
            if isdefined(self.inputs.bzero):
                suffix = 'b0'
            else:
                suffix = 'dw'
            filename = os.path.join(
                os.getcwd(), "{}_{}{}".format(base, suffix, ext))
        return filename


# =============================================================================
# MR Convert
# =============================================================================


class MRConvertInputSpec(MRTrix3BaseInputSpec):
    in_file = traits.Either(
        File(exists=True, desc="Input file"),
        Directory(exists=True, desc="Input directory (assumed to be DICOM)"),
        mandatory=True, argstr='%s', position=-2)
    out_file = File(
        genfile=True, argstr='%s', position=-1, hash_files=False,
        desc=("Output (converted) file. If no path separators (i.e. '/' on "
              "*nix) are found in the provided output file then the CWD (when "
              "the workflow is run, i.e. the working directory) will be "
              "prepended to the output path."))
    out_ext = traits.Str(
        mandatory=False,
        desc=("The extension (and therefore the file format) to use when the "
              "output file path isn't provided explicitly"))
    coord = traits.Tuple(
        traits.Int(), traits.Int(),
        mandatory=False, argstr='-coord %d %d',
        desc=("extract data from the input image only at the coordinates "
              "specified."))
    vox = traits.Str(
        mandatory=False, argstr='-vox %s',
        desc=("change the voxel dimensions of the output image. The new sizes "
              "should be provided as a comma-separated list of values. Only "
              "those values specified will be changed. For example: 1,,3.5 "
              "will change the voxel size along the x & z axes, and leave the "
              "y-axis voxel size unchanged."))
    axes = traits.Str(
        mandatory=False, argstr='-axes %s',
        desc=("specify the axes from the input image that will be used to form"
              " the output image. This allows the permutation, ommission, or "
              "addition of axes into the output image. The axes should be "
              "supplied as a comma-separated list of axes. Any ommitted axes "
              "must have dimension 1. Axes can be inserted by supplying -1 at "
              "the corresponding position in the list."))
    scaling = traits.Str(
        mandatory=False, argstr='-scaling %s',
        desc=("specify the data scaling parameters used to rescale the "
              "intensity values. These take the form of a comma-separated "
              "2-vector of floating-point values, corresponding to offset & "
              "scale, with final intensity values being given by offset + "
              "scale * stored_value. By default, the values in the input "
              "image header are passed through to the output image header "
              "when writing to an integer image, and reset to 0,1 (no scaling)"
              " for floating-point and binary images. Note that his option has"
              " no effect for floating-point and binary images."))
    stride = traits.Str(
        mandatory=False, argstr='-stride %s',
        desc=("specify the strides of the output data in memory, as a "
              "comma-separated list. The actual strides produced will depend "
              "on whether the output image format can support it."))
    dataset = traits.Str(
        mandatory=False, argstr='-dataset %s',
        desc=("specify output image data type. Valid choices are: float32, "
              "float32le, float32be, float64, float64le, float64be, int64, "
              "uint64, int64le, uint64le, int64be, uint64be, int32, uint32, "
              "int32le, uint32le, int32be, uint32be, int16, uint16, int16le, "
              "uint16le, int16be, uint16be, cfloat32, cfloat32le, cfloat32be, "
              "cfloat64, cfloat64le, cfloat64be, int8, uint8, bit."))
    export_grad_mrtrix = traits.Str(
        mandatory=False, argstr='-export_grad_mrtrix %s',
        desc=("export the diffusion-weighted gradient table to file in MRtrix "
              "format"))
    export_grad_fsl = traits.Str(
        mandatory=False, argstr='-export_grad_fsl %s',
        desc=("export the diffusion-weighted gradient table to files in FSL "
              "(bvecs / bvals) format"))
    quiet = traits.Bool(
        mandatory=False, argstr="-quiet",
        description="Don't display output during conversion")


class MRConvertOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='Extracted encoding gradients')


class MRConvert(MRTrix3Base):

    _cmd = 'mrconvert'
    input_spec = MRConvertInputSpec
    output_spec = MRConvertOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = self._gen_outfilename()
        return outputs

    def _gen_filename(self, name):
        if name == 'out_file':
            gen_name = self._gen_outfilename()
        else:
            assert False
        return gen_name

    def _gen_outfilename(self):
        if isdefined(self.inputs.out_file):
            out_name = self.inputs.out_file
        else:
            base, orig_ext = split_extension(
                os.path.basename(self.inputs.in_file))
            ext = (self.inputs.out_ext
                   if isdefined(self.inputs.out_ext) else orig_ext)
            out_name = os.path.join(os.getcwd(),
                                    "{}_conv{}".format(base, ext))
        return out_name


class DWIPreprocInputSpec(CommandLineInputSpec):
    in_file = traits.File(
        mandatory=True, argstr='%s',
        desc=("The input DWI series to be corrected"), position=-2)
    out_file = File(
        genfile=True, argstr='%s', position=-1, hash_files=False,
        desc="Output preprocessed filename")
    rpe_pair = traits.Bool(
        mandatory=False, argstr="-rpe_pair",
        desc=("forward reverse Provide a pair of images to use for "
              "inhomogeneity field estimation; note that the FIRST of these "
              "two images must have the same phase"))
    rpe_header = traits.Bool(
        mandatory=False, argstr="-rpe_header",
        description=(
            "Attempt to read the phase-encoding information from headeer"))
    # Arguments
    pe_dir = traits.Str(
        argstr='-pe_dir %s', desc=(
            "The phase encode direction; can be a signed axis "
            "number (e.g. -0, 1, +2) or a code (e.g. AP, LR, IS)"))
    se_epi = traits.File(
        argstr='-se_epi %s',
        desc=("forward and reverse pair concanenated into a single 4D image "
              "for inhomogeneity field estimation; note that the FIRST of "
              "these two images must have the same phase"),
        position=1)


class DWIPreprocOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='Pre-processed DWI dataset')


class DWIPreproc(CommandLine):

    _cmd = 'dwipreproc'
    input_spec = DWIPreprocInputSpec
    output_spec = DWIPreprocOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = self._gen_outfilename()
        return outputs

    def _gen_filename(self, name):
        if name == 'out_file':
            gen_name = self._gen_outfilename()
        else:
            assert False
        return gen_name

    def _gen_outfilename(self):
        if isdefined(self.inputs.out_file):
            out_name = self.inputs.out_file
        else:
            base, ext = split_extension(
                os.path.basename(self.inputs.in_file))
            out_name = os.path.join(
                os.getcwd(), "{}_preproc{}".format(base, ext))
        return out_name


class DWI2MaskInputSpec(CommandLineInputSpec):
    # Arguments
    bvalue_scaling = File(
        mandatory=False, argstr='-bvalue_scaling %s',
        desc=("specifies whether the b-values should be scaled by the square "
              "of the corresponding DW gradient norm, as often required for "
              "multi-shell or DSI DW acquisition schemes. The default action "
              "can also be set in the MRtrix config file, under the "
              "BValueScaling entry. Valid choices are yes/no, true/false, "
              "0/1 (default: true)."))
    in_file = traits.File(
        mandatory=True, argstr='%s', exists=True,
        desc=("The input DWI series to be corrected"), position=-2)
    out_file = File(
        genfile=True, argstr='%s', position=-1, hash_files=False,
        desc="Output preprocessed filename")
    grad = traits.Str(
        mandatory=False, argstr='-grad %s',
        desc=("specify the diffusion-weighted gradient scheme used in the  "
              "acquisition. The program will normally attempt to use the  "
              "encoding stored in the image header. This should be supplied  "
              "as a 4xN text file with each line is in the format [ X Y Z b ],"
              " where [ X Y Z ] describe the direction of the applied  "
              "gradient, and b gives the b-value in units of s/mm^2."))

    fslgrad = traits.Tuple(
        File(exists=True, desc="gradient directions file (bvec)"),  # @UndefinedVariable @IgnorePep8
        File(exists=True, desc="b-values (bval)"),  # @UndefinedVariable @IgnorePep8
        argstr='-fslgrad %s %s', mandatory=False,
        desc=("specify the diffusion-weighted gradient scheme used in the "
              "acquisition in FSL bvecs/bvals format."))


class DWI2MaskOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='Bias-corrected DWI dataset')


class DWI2Mask(CommandLine):

    _cmd = 'dwi2mask'
    input_spec = DWI2MaskInputSpec
    output_spec = DWI2MaskOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = self._gen_outfilename()
        return outputs

    def _gen_filename(self, name):
        if name == 'out_file':
            gen_name = self._gen_outfilename()
        else:
            assert False
        return gen_name

    def _gen_outfilename(self):
        if isdefined(self.inputs.out_file):
            out_name = self.inputs.out_file
        else:
            base, ext = split_extension(
                os.path.basename(self.inputs.in_file))
            out_name = os.path.join(
                os.getcwd(), "{}_biascorrect{}".format(base, ext))
        return out_name


class DWIBiasCorrectInputSpec(CommandLineInputSpec):
    # Arguments
    mask = File(
        mandatory=True, argstr='-mask %s',
        desc=("Whole brain mask"))
    method = traits.Str(
        mandatory=False, argstr='-%s',
        desc=("Method used to correct for biases (either 'fsl' or 'ants')"))
    bias = File(
        mandatory=False, argstr='-bias %s',
        desc=("Output the estimated bias field"))
    in_file = traits.Str(
        mandatory=True, argstr='%s',
        desc=("The input DWI series to be corrected"), position=-2)
    out_file = File(
        genfile=True, argstr='%s', position=-1, hash_files=False,
        desc="Output preprocessed filename")
    grad = traits.Str(
        mandatory=False, argstr='-grad %s',
        desc=("specify the diffusion-weighted gradient scheme used in the  "
              "acquisition. The program will normally attempt to use the  "
              "encoding stored in the image header. This should be supplied  "
              "as a 4xN text file with each line is in the format [ X Y Z b ],"
              " where [ X Y Z ] describe the direction of the applied  "
              "gradient, and b gives the b-value in units of s/mm^2."))

    fslgrad = traits.Tuple(
        File(exists=True, desc="gradient directions file (bvec)"),  # @UndefinedVariable @IgnorePep8
        File(exists=True, desc="b-values (bval)"),  # @UndefinedVariable @IgnorePep8
        argstr='-fslgrad %s %s', mandatory=False,
        desc=("specify the diffusion-weighted gradient scheme used in the "
              "acquisition in FSL bvecs/bvals format."))


class DWIBiasCorrectOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='Bias-corrected DWI dataset')


class DWIBiasCorrect(CommandLine):

    _cmd = 'dwibiascorrect'
    input_spec = DWIBiasCorrectInputSpec
    output_spec = DWIBiasCorrectOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = self._gen_outfilename()
        return outputs

    def _gen_filename(self, name):
        if name == 'out_file':
            gen_name = self._gen_outfilename()
        else:
            assert False
        return gen_name

    def _gen_outfilename(self):
        if isdefined(self.inputs.out_file):
            out_name = self.inputs.out_file
        else:
            base, ext = split_extension(
                os.path.basename(self.inputs.in_file))
            out_name = os.path.join(
                os.getcwd(), "{}_biascorrect{}".format(base, ext))
        return out_name


class MRCatInputSpec(CommandLineInputSpec):

    first_scan = traits.File(
        exists=True, mandatory=True, desc="First input image", argstr="%s",
        position=-3)

    second_scan = traits.File(
        exists=True, mandatory=True, desc="Second input image", argstr="%s",
        position=-2)

    out_file = traits.File(
        genfile=True, desc="Output filename", position=-1, hash_files=False,
        argstr="%s")

    axis = traits.Int(
        desc="The axis along which the scans will be concatenated",
        argstr="-axis %s")

    quiet = traits.Bool(
        mandatory=False, argstr="-quiet",
        description="Don't display output during concatenation")


class MRCatOutputSpec(TraitedSpec):

    out_file = File(exists=True, desc='Pre-processed DWI dataset')


class MRCat(CommandLine):

    _cmd = 'mrcat'
    input_spec = MRCatInputSpec
    output_spec = MRCatOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = self._gen_outfilename()
        return outputs

    def _gen_filename(self, name):
        if name == 'out_file':
            gen_name = self._gen_outfilename()
        else:
            assert False
        return gen_name

    def _gen_outfilename(self):
        if isdefined(self.inputs.out_file):
            out_name = self.inputs.out_file
        else:
            first, ext = split_extension(
                os.path.basename(self.inputs.first_scan))
            second, _ = split_extension(
                os.path.basename(self.inputs.second_scan))
            out_name = os.path.join(
                os.getcwd(), "{}_{}_concat{}".format(first, second, ext))
        return out_name


# =============================================================================
# DWI Denoise
# =============================================================================


class DWIDenoiseInputSpec(CommandLineInputSpec):
    in_file = traits.Either(
        File(exists=True, desc="Input file"),
        Directory(exists=True, desc="Input directory (assumed to be DICOM)"),
        mandatory=True, argstr='%s', position=-2)
    out_file = File(
        genfile=True, argstr='%s', position=-1, hash_files=False,
        desc=("Output (converted) file. If no path separators (i.e. '/' on "
              "*nix) are found in the provided output file then the CWD (when "
              "the workflow is run, i.e. the working directory) will be "
              "prepended to the output path."))
    noise = File(
        genfile=True, argstr="-noise %s",
        desc=("The estimated spatially-varying noise level"))
    mask = File(
        argstr="-mask %s",
        desc=("Perform the de-noising in the specified mask"))
    extent = traits.Tuple(
        traits.Int(), traits.Int(), traits.Int(), argstr="-extent %d,%d,%d",
        desc="Extent of the kernel")


class DWIDenoiseOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='De noised image')
    noise = File(desc=("The estimated spatially-varying noise level"))


class DWIDenoise(CommandLine):

    _cmd = 'dwidenoise'
    input_spec = DWIDenoiseInputSpec
    output_spec = DWIDenoiseOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = self._gen_outfname()
        outputs['noise'] = self._gen_noisefname()
        return outputs

    def _gen_filename(self, name):
        if name == 'out_file':
            gen_name = self._gen_outfname()
        elif name == 'noise':
            gen_name = self._gen_noisefname()
        else:
            assert False
        return gen_name

    def _gen_outfname(self):
        if isdefined(self.inputs.out_file):
            out_name = self.inputs.out_file
        else:
            base, ext = split_extension(
                os.path.basename(self.inputs.in_file))
            out_name = os.path.join(os.getcwd(),
                                    "{}_conv{}".format(base, ext))
        return out_name

    def _gen_noisefname(self):
        if isdefined(self.inputs.out_file):
            out_name = self.inputs.out_file
        else:
            base, ext = split_extension(
                os.path.basename(self.inputs.in_file))
            out_name = os.path.join(os.getcwd(),
                                    "{}_noise{}".format(base, ext))
        return out_name


class DWIIntensityNormInputSpec(MRTrix3BaseInputSpec):

    in_files = InputMultiPath(
        File(exists=True),
        desc="The input DWI images to normalize",
        mandatory=True)

    masks = InputMultiPath(
        File(exists=True),
        desc=("Input directory containing brain masks, corresponding to "
              "one per input image in the same order"),
        mandatory=True)

    fa_threshold = traits.Float(
        argstr='-fa_threshold %s',
        desc=("The threshold applied to the Fractional Anisotropy group "
              "template used to derive an approximate white matter mask"))

    fa_template = File(
        genfile=True, hash_files=False,
        desc=("The output population specific FA template, which is "
              "threshold to estimate a white matter mask"),
        argstr='%s', position=-2)

    wm_mask = File(
        genfile=True, hash_files=False,
        desc=("Input directory containing brain masks, corresponding to "
              "one per input image (with the same file name prefix"),
        argstr='%s', position=-1)

    in_dir = Directory(
        genfile=True,
        desc="The input directory to collate the DWI images within",
        argstr='%s', position=-5)

    mask_dir = File(
        genfile=True,
        desc=("Input directory to collate the brain masks within"),
        argstr='%s', position=-4)

    out_dir = Directory(
        genfile=True, argstr='%s', position=-3, hash_files=False,
        desc=("The output directory to containing the normalised DWI images"))


class DWIIntensityNormOutputSpec(TraitedSpec):

    out_files = traits.List(
        File(exists=True),
        desc=("The intensity normalised DWI images"))

    fa_template = File(
        exists=True,
        desc=("The output population specific FA templates, which is "
              "threshold to estimate a white matter mask"))

    wm_mask = File(
        exists=True,
        desc=("Input directory containing brain masks, corresponding to "
              "one per input image (with the same file name prefix"))


class DWIIntensityNorm(MRTrix3Base):

    _cmd = 'dwiintensitynorm'
    input_spec = DWIIntensityNormInputSpec
    output_spec = DWIIntensityNormOutputSpec

    def _run_interface(self, *args, **kwargs):
        self._link_into_dir(self.inputs.in_files, self._gen_in_dir_name())
        self._link_into_dir(self.inputs.masks, self._gen_mask_dir_name())
        return super(DWIIntensityNorm, self)._run_interface(*args, **kwargs)

    def _list_outputs(self):
        outputs = self.output_spec().get()
        out_dir = self._gen_out_dir_name()
        outputs['out_files'] = sorted(
            os.path.join(out_dir, f) for f in os.listdir(out_dir))
        outputs['fa_template'] = self._gen_fa_template_name()
        outputs['wm_mask'] = self._gen_wm_mask_name()
        return outputs

    def _gen_filename(self, name):
        if name == 'in_dir':
            gen_name = self._gen_in_dir_name()
        elif name == 'mask_dir':
            gen_name = self._gen_mask_dir_name()
        elif name == 'out_dir':
            gen_name = self._gen_out_dir_name()
        elif name == 'fa_template':
            gen_name = self._gen_fa_template_name()
        elif name == 'wm_mask':
            gen_name = self._gen_wm_mask_name()
        else:
            assert False
        return gen_name

    def _gen_in_dir_name(self):
        if isdefined(self.inputs.in_dir):
            out_name = self.inputs.in_dir
        else:
            out_name = os.path.join(os.getcwd(), 'in_dir')
        return out_name

    def _gen_mask_dir_name(self):
        if isdefined(self.inputs.mask_dir):
            out_name = self.inputs.mask_dir
        else:
            out_name = os.path.join(os.getcwd(), 'mask_dir')
        return out_name

    def _gen_out_dir_name(self):
        if isdefined(self.inputs.out_dir):
            out_name = self.inputs.out_dir
        else:
            out_name = os.path.join(os.getcwd(), 'out_dir')
        return out_name

    def _gen_fa_template_name(self):
        if isdefined(self.inputs.fa_template):
            out_name = self.inputs.fa_template
        else:
            ext = split_extension(self.inputs.in_files[0])[1]
            out_name = os.path.join(os.getcwd(), 'fa_template' + ext)
        return out_name

    def _gen_wm_mask_name(self):
        if isdefined(self.inputs.wm_mask):
            out_name = self.inputs.wm_mask
        else:
            ext = split_extension(self.inputs.in_files[0])[1]
            out_name = os.path.join(os.getcwd(), 'wm_mask' + ext)
        return out_name

    def _link_into_dir(self, fpaths, dirpath):
        """
        Symlinks the given file paths into the given directory, making the
        directory if necessary
        """
        try:
            os.makedirs(dirpath)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        num_digits = int(math.ceil(math.log(len(fpaths), 10)))
        for i, fpath in enumerate(fpaths):
            _, ext = split_extension(fpath)
            os.symlink(fpath,
                       os.path.join(dirpath, str(i).zfill(num_digits) + ext))
