import os
import os.path as op
from nipype.interfaces.base import File, traits, Directory
from arcana.utils import parse_value, split_extension
from .matlab import BaseMatlab, BaseMatlabInputSpec, BaseMatlabOutputSpec


class GrappaInputSpec(BaseMatlabInputSpec):
    in_file = File(
        exists=True, mandatory=True,
        desc="The data file of the 'custom_kspace_format'")
    ref_file = File(
        exists=True, mandatory=True,
        desc="The reference file of the 'custom_kspace_format'")
    hdr_file = File(
        exists=True, mandatory=True,
        desc=("The header file of the 'custom_kspace_format'"))

    acceleration = traits.Int(2, usedefault=True,
                              desc="The acceleration factor to use")


class GrappaOutputSpec(BaseMatlabOutputSpec):

    channels_dir = Directory(exists=True)
    main_field_strength = traits.Float(
        desc="The orientation of the scanners main magnetic field"),
    main_field_orient = traits.Tuple(
        (traits.Int, traits.Int, traits.Int),
        desc="The orientation of the scanners main magnetic field")
    echo_times = traits.Tuple(traits.Int, desc="The echo times in seconds")
    larmor_freq = traits.Float(desc="The larmor frequency of the scanner")


class Grappa(BaseMatlab):
    """
    Performs Grappa reconstruction on the input k-space file and saves a
    magnitude image (sum of squares over coils and echos) to 'out_file' and
    real and imaginary components of each coil per channel to 'channels_dir'.
    It also extracts header fields from the 'matlab_kspace_format' file so they
    can be saved as fields in the repository.
    """

    input_spec = GrappaInputSpec
    output_spec = GrappaOutputSpec

    def script(self, **inputs):
        """
        Generate script to run Grappa reconstruction for each echo of each
        channel
        """
        script = ("recon_grappa2('{data_file}', '{ref_file}', '{hdr_file}', "
                  "'{mag_file}', '{channels_dir}', {rpe}, 0);"
                  .format(
                      data_file=self.inputs.in_file,
                      ref_file=self.inputs.ref_file,
                      hdr_file=self.inputs.hdr_file,
                      mag_file=self.out_file,
                      channels_dir=self.channels_dir,
                      rpe=self.inputs.acceleration))
        return script

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = self.out_file
        outputs['channels_dir'] = self.channels_dir
        return outputs

    @property
    def channels_dir(self):
        channels_dir = op.realpath(op.abspath(op.join(self.work_dir,
                                                      'channels')))
        if not op.exists(channels_dir):
            os.makedirs(channels_dir)
        return channels_dir

    @property
    def out_file(self):
        return op.realpath(op.abspath(
            op.join(self.work_dir, 'out_file.nii.gz')))
