import os.path
from nipype.interfaces.base import (
    traits, InputMultiPath, File, TraitedSpec, isdefined)
from nipype.interfaces.mrtrix3.reconst import (
    MRTrix3Base, MRTrix3BaseInputSpec)
from arcana.utils import split_extension


class MRResizeInputSpec(MRTrix3BaseInputSpec):
    in_file = File(exists=True, argstr='%s', mandatory=True, position=-2,
                   desc='input files')
    out_file = File(genfile=True, argstr='%s', desc=(""), position=-1)

    size = traits.List(
        traits.Int(),
        mandatory=False, argstr='-size %s',
        desc=("define the new image size for the output image. "
              "This should be specified as a comma-separated "
              "list."))

    voxel = traits.List(
        traits.Float(),
        mandatory=False, argstr='-voxel %s',
        desc=("define the new voxel size for the output image."
              "This can be specified either as a single value "
              "to be used for all dimensions, or as a "
              "comma-separated list of the size for each voxel"
              "dimension."))

    scale = traits.Float(
        mandatory=False, argstr='-scale %s',
        desc=("scale the image resolution by the supplied factor."
              " This can be specified either as a single value to"
              " be used for all dimensions, or as a "
              "comma-separated list of scale factors for each "
              "dimension."))

    interp = traits.Enum(
        'nearest', 'linear', 'cubic', 'sinc', default='cubic', mandatory=False,
        argstr='-interp %s',
        desc=("set the interpolation method to use when resizing"
              " (choices: nearest, linear, cubic, sinc. Default:"
              " cubic)."))


class MRResizeOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc=("The output file"))


class MRResize(MRTrix3Base):
    """
    Note that if the image is 4D, then only the first 3 dimensions can be
     resized.

    Also note that if the image is down-sampled, the appropriate smoothing is
    automatically applied using Gaussian smoothing.
    """

    _cmd = "mrresize"
    input_spec = MRResizeInputSpec
    output_spec = MRResizeOutputSpec

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
                os.getcwd(), "{}_resize{}".format(base, ext))
        return out_name


class MRRegisterInputSpec(MRTrix3BaseInputSpec):
    in_file = File(exists=True, argstr='%s', mandatory=True, position=-3,
                   desc='input files')
    reference = File(exists=True, argstr='%s', mandatory=True, position=-2,
                     desc="The reference image to register to")
    out_file = File(genfile=True, argstr='%s', desc=(""), position=-1)


class MRRegisterOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc=(""))


class MRRegister(MRTrix3Base):
    """MRRegister"""

    _cmd = "mrregister"
    input_spec = MRRegisterInputSpec
    output_spec = MRRegisterOutputSpec

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
                os.getcwd(), "{}_register{}".format(base, ext))
        return out_name


class MRThresholdInputSpec(MRTrix3BaseInputSpec):
    in_file = File(exists=True, argstr='%s', mandatory=True, position=-3,
                   desc='input files')
    out_file = File(genfile=True, argstr='%s', desc=(""), position=-1)


class MRThresholdOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc=(""))


class MRThreshold(MRTrix3Base):
    """MRThreshold"""

    _cmd = "mrthreshold"
    input_spec = MRThresholdInputSpec
    output_spec = MRThresholdOutputSpec

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
                os.getcwd(), "{}_threshold{}".format(base, ext))
        return out_name


class FixelReorientInputSpec(MRTrix3BaseInputSpec):
    in_file = File(exists=True, argstr='%s', mandatory=True, position=-3,
                   desc='input files')
    out_file = File(genfile=True, argstr='%s', desc=(""), position=-1)


class FixelReorientOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc=(""))


class FixelReorient(MRTrix3Base):
    """FixelReorient"""

    _cmd = "fixelreorient"
    input_spec = FixelReorientInputSpec
    output_spec = FixelReorientOutputSpec

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
                os.getcwd(), "{}_reorient{}".format(base, ext))
        return out_name
