import os
import os.path as op
from nipype.interfaces.base import File, traits, Directory
from arcana.utils import parse_value
from .matlab import BaseMatlab, BaseMatlabInputSpec, BaseMatlabOutputSpec


class GrappaInputSpec(BaseMatlabInputSpec):
    in_file = File(
        exists=True, mandatory=True,
        desc="Input file in custom matlab format (see matlab_kspace_format)")
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

    out_sep = '#####'

    def script(self, **inputs):
        """
        Generate script to run Grappa reconstruction for each echo of each
        channel
        """
        script = """
            S = load('{in_file}');
            [img_recon_ch, smaps] = recon_grappa2(...
                S.calib_scan, S.data_scan, S.dims(2), S.dims(3), 0, {rpe});
            % Calculate combined magnitude, and real and imaginary images per
            % channel and save to nifti files
            mag = squeeze(sqrt(sum(sum(abs(img_recon_ch).^2, 1), 5)));
            out_nii = make_nii(mag, voxel_size, [], [],...
                             'Sum of squares magnitude average across echos');
            save_nii(out_nii, '{out_file}');

            for i=1:size(img_recon_ch, 1)
                coil = squeeze(img_recon_ch(i, :, :, :, :));
                out_nii = make_nii(real(coil), voxel_size, [], [],...
                                   'Real image per coil');
                save_nii(out_nii, sprintf('%s%sReal_c%d.nii.gz',...
                                          '{channels_dir}', filesep, i));

                out_nii = make_nii(imag(coil), voxel_size, [], [],...
                                   'Imaginary image per coil');
                save_nii(out_nii, sprintf('%s%sImaginary_c%d.nii.gz',...
                                          '{channels_dir}', filesep, i));
            end
            % Print out header values so they can be output into traits
            fprintf('{out_sep}\\n');
            fprintf('echo_times=');
            for i=1:length(S.TE)
                if i ~= 1
                    fprintf(', ');
                end
                fprintf('%f', S.TE(i));
            end
            fprintf('\\n');
            fprintf('main_field_strength=%f\\n', S.B0_strength);
            fprintf('main_field_orient=%f, %f, %f\\n',...
                    S.B0_dir(1), S.B0_dir(2), S.B0_dir(3));
            fprintf('larmor_freq=%f\\n', S.larmor_frequency);
            """.format(in_file=self.inputs.in_file,
                       out_file=self.out_file,
                       out_sep=self.out_sep,
                       channels_dir=self.channels_dir,
                       rpe=self.inputs.acceleration)
        return script

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = self.out_file
        outputs['channels_dir'] = self.channels_dir
        for field in outputs['raw_output'].split(self.out_sep)[1].split('\n'):
            name, val = field.split('=')
            val = parse_value(val)
            if isinstance(val, list):
                val = tuple(val)
            outputs[name] = val
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
