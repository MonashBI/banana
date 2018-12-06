import os.path
import errno
import math
from nipype.interfaces.base import (
    File, Directory, TraitedSpec, isdefined, traits, InputMultiPath)
from nipype.interfaces.mrtrix3.reconst import (
    MRTrix3Base, MRTrix3BaseInputSpec)
from arcana.utils import split_extension


# TODO: Write MRtrixBaseInputSpec with all the generic parameters included

# =============================================================================
# Extract MR gradients
# =============================================================================

class DWIPreprocInputSpec(MRTrix3BaseInputSpec):
    in_file = traits.File(
        mandatory=True, argstr='%s',
        desc=("The input DWI series to be corrected"), position=1)
    out_file = File(
        genfile=True, argstr='%s', position=2, hash_files=False,
        desc="Output preprocessed filename")
    out_file_ext = traits.Str(
        desc='Specify the extention for the final output')
    rpe_none = traits.Bool(
        mandatory=False, argstr='-rpe_none',
        desc=("No reverse phase encoded reference image provided"))
    rpe_pair = traits.Bool(
        mandatory=False, argstr="-rpe_pair",
        desc=("Provide a pair of images to use for "
              "inhomogeneity field estimation; note that the FIRST of these "
              "two images must have the same phase"))
    rpe_header = traits.Bool(
        mandatory=False, argstr="-rpe_header",
        desc=(
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
        position=3)
    eddy_parameters = traits.Str(
        argstr='-eddy_options "%s"', desc='parameters to be passed to eddy')
    no_clean_up = traits.Bool(True, argstr='-nocleanup',
                              desc='Do not delete the temporary folder')
    temp_dir = Directory(genfile=True, argstr='-tempdir %s',
                         desc="Specify the temporay directory")


class DWIPreprocOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='Pre-processed DWI dataset')
    eddy_parameters = File(desc='File with eddy parameters')


class DWIPreproc(MRTrix3Base):

    _cmd = 'dwipreproc'
    input_spec = DWIPreprocInputSpec
    output_spec = DWIPreprocOutputSpec

    def _list_outputs(self):

        outputs = self.output_spec().get()
        outputs['out_file'] = os.path.abspath(self._gen_outfilename())
        outputs['eddy_parameters'] = os.path.abspath(os.path.join(
            self._gen_tempdir(), 'dwi_post_eddy.eddy_parameters'))
        return outputs

    def _gen_filename(self, name):
        if name == 'out_file':
            gen_name = self._gen_outfilename()
        elif name == 'temp_dir':
            gen_name = self._gen_tempdir()
        else:
            assert False
        return gen_name

    def _gen_outfilename(self):
        if isdefined(self.inputs.out_file):
            out_name = self.inputs.out_file
        else:
            base, ext = split_extension(
                os.path.basename(self.inputs.in_file))
            if isdefined(self.inputs.out_file_ext):
                extension = self.inputs.out_file_ext
            else:
                extension = ext
            out_name = "{}_preproc{}".format(base, extension)
        return out_name

    def _gen_tempdir(self):
        if isdefined(self.inputs.temp_dir):
            temp_dir = self.inputs.temp_dir
        else:
            base, _ = split_extension(
                os.path.basename(self.inputs.in_file))
            temp_dir = "{}_tempdir".format(base)
        return temp_dir


class DWI2MaskInputSpec(MRTrix3BaseInputSpec):
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


class DWI2MaskOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='Bias-corrected DWI dataset')


class DWI2Mask(MRTrix3Base):

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


class DWIBiasCorrectInputSpec(MRTrix3BaseInputSpec):
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


class DWIBiasCorrectOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='Bias-corrected DWI dataset')


class DWIBiasCorrect(MRTrix3Base):

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

# =============================================================================
# DWI Denoise
# =============================================================================


class DWIDenoiseInputSpec(MRTrix3BaseInputSpec):
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
    out_file_ext = traits.Str(desc='Extention of the output file.')
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


class DWIDenoise(MRTrix3Base):

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
            if isdefined(self.inputs.out_file_ext):
                ext = self.inputs.out_file_ext
                base, _ = split_extension(os.path.basename(
                    self.inputs.in_file))
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
            if isdefined(self.inputs.out_file_ext):
                ext = self.inputs.out_file_ext
                base, _ = split_extension(os.path.basename(
                    self.inputs.in_file))
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
