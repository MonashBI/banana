import os
import os.path as op
from nipype.interfaces.base import File, traits
from .matlab import BaseMatlab, BaseMatlabInputSpec, BaseMatlabOutputSpec


class GrappaInputSpec(BaseMatlabInputSpec):
    in_file = File(
        exists=True, mandatory=True,
        desc="Input file in custom matlab format (see matlab_kspace_format)")
    acceleration = traits.Int(2, usedefault=True,
                              desc="The acceleration factor to use")


class GrappaOutputSpec(BaseMatlabOutputSpec):

    sensitivity_maps = File(exists=True)


class Grappa(BaseMatlab):
    """
    Reads a Siemens TWIX (multi-channel k-space) file and saves it in a Matlab
    file in 'matlab_kspace' format (see banana.file_format for details)
    """

    input_spec = GrappaInputSpec
    output_spec = BaseMatlabOutputSpec

    def script(self, **inputs):
        """
        Generate script to run Grappa reconstruction for each echo of each
        channel
        """
        script = """
            S = load({in_file});
            [img_recon_ch, smaps] = recon_grappa2(
                S.calib_scan, S.data_scan, S.dims(2), S.dims(3), 0, {rpe});

            % Calculate combined magnitude, and real and imaginary images per
            % channel
            mag = squeeze(sqrt(sum(abs(Img_recon_ch).^2,[1 5])));
            out_nii = make_nii(mag, voxel_size, [], [],...
                              'Sum of squares magnitude average across echos');
            save_nii(out_nii, {out_file});

            for i=1:size(Img_recon_ch,1)
                coil = squeeze(Img_recon_ch(i, :, :, :, :));
                out_nii = make_nii(real(coil), voxel_size, [], [],...
                                   'Real image per coil');
                save_nii(out_nii, sprintf('%s%sReal_c%d.nii.gz',...
                                          {channels_dir}, filesep, i));

                out_nii = make_nii(imag(coil), voxel_size, [], [],...
                                   'Imaginary image per coil');
                save_nii(out_nii, sprintf('%s%sImaginary_c%d.nii.gz',...
                                          {channels_dir}, filesep, i));
            end
            """.format(in_file=self.inputs.in_file,
                       out_file=self.out_file,
                       channels_dir=self.channels_dir,
                       rpe=self.inputs.acceleration)
        return script

    def _list_outputs(self):
        outputs = super()._list_outputs()
        outputs['channels_dir'] = self.channels_dir
        return outputs

    @property
    def magnitude_file(self):
        return op.realpath(op.abspath(op.join(self.work_dir,
                                              'magnitude.nii.gz')))

    @property
    def channels_dir(self):
        channels_dir = op.realpath(op.abspath(op.join(self.work_dir,
                                                      'channels')))
        if not op.exists(channels_dir):
            os.makedirs(channels_dir)
        return channels_dir

    @property
    def sensitivity_maps(self):
        return op.realpath(
            op.abspath(op.join(self.work_dir, 'sensitivity_maps.nii')))
