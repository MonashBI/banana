import os
import os.path as op
from copy import deepcopy
from collections import defaultdict
import logging
import re
import numpy as np
import nibabel as nib
from nipype.interfaces.base import (
    TraitedSpec, traits, BaseInterface, BaseInterfaceInputSpec, File,
    Directory, isdefined)
from banana.exceptions import BananaUsageError
logger = logging.getLogger('banana')


class ToPolarCoordsInputSpec(BaseInterfaceInputSpec):
    in_dir = Directory(exists=True, mandatory=True)
    in_fname_re = traits.Str(
        r'(?P<axis>[a-z]+)_(?P<channel>\d+)\.nii\.gz',
        usedefault=True, desc=(
            "Regex to extract the channel, echo and axis "
            "(i.e. real or imaginary) information from the input file name. "
            "Must incluce named groups for 'channel', 'echo' and 'axis'"))
    out_fname_str = traits.Str(
        'coil_{channel}_{echo}.nii.gz', usedefault=True,
        desc=("The format string used to generate the save channel filenames. "
              "Must use the 'channel' and 'echo' field names"))
    real_label = traits.Str('REAL', usedefault=True, desc=(
        "The label used to specify the real component image"))
    imaginary_label = traits.Str('IMAGINARY', usedefault=True, desc=(
        "The label used to specify the real component image"))
    combined_dir = Directory(genfile=True, desc=(
        "Output directory for coil magnitude and phase images. "
        "Files will be saved with the name "
        "'Raw_Coil_<channel>_<echo>.nii.gz'"))
    magnitudes_dir = Directory(genfile=True, desc=(
        "Output directory for coil magnitude images."))
    phases_dir = Directory(genfile=True, desc=(
        "Output directory for coil phase images"))


class ToPolarCoordsOutputSpec(TraitedSpec):
    combined_images = traits.List(
        File(exists=True),
        desc="List of combined images for each echo using least squares")
    first_echo = File(exists=True,
                      desc="The first echo of the combined images")
    last_echo = File(exists=True,
                     desc="The last echo of the combined images")
    coil_magnitudes = traits.List(
        traits.List(File(exists=True)),
        desc=("List of magnitude images for each coil for each echo"))
    coil_phases = traits.List(
        traits.List(File(exists=True)),
        desc=("List of magnitude images for each coil for each echo"))
    combined_dir = Directory(exists=True, desc=(
        "Output directory for combined magnitude images for each echo time "))
    magnitudes_dir = Directory(exists=True, desc=(
        "Output directory for coil magnitude images"))
    phases_dir = Directory(exists=True, desc=(
        "Output directory for coil phase images"))


class ToPolarCoords(BaseInterface):
    """
    Takes all REAL and IMAGINARY pairs in current directory and prepares
    them for Phase and QSM processing.

    1. Existence of pairs is checked
    2. Files are load/save cycled for formatting and rename for consistency
    3. Magnitude and Phase components are produced
    4. Coils are combined for single magnitude images per echo
    """
    input_spec = ToPolarCoordsInputSpec
    output_spec = ToPolarCoordsOutputSpec

    def _run_interface(self, runtime):
        return runtime

    def _list_outputs(self):
        print("in-dir: " + self.inputs.in_dir)
        outputs = self._outputs().get()
        # Get names for output directories
        combined_dir = outputs['combined_dir'] = self._gen_filename(
            'combined_dir')
        mags_dir = outputs['magnitudes_dir'] = self._gen_filename(
            'magnitudes_dir')
        phases_dir = outputs['phases_dir'] = self._gen_filename(
            'phases_dir')
        # Ensure output directories exist
        os.makedirs(combined_dir, exist_ok=True)
        os.makedirs(mags_dir, exist_ok=True)
        os.makedirs(phases_dir, exist_ok=True)
        outputs['combined_images'] = []
        coil_mags = outputs['coil_magnitudes'] = []
        coil_phases = outputs['coil_phases'] = []
        # A default dict with three levels of keys to hold the file names
        # sorted into echo, channel and complex axis
        paths = defaultdict(dict)
        # Compile regular expression for extracting channel, echo and
        # complex axis indices from input file names
        fname_re = re.compile(self.inputs.in_fname_re)
        for fname in os.listdir(self.inputs.in_dir):
            match = fname_re.match(fname)
            if match is None:
                logger.warning("Skipping '{}' file in '{}' as it doesn't "
                               "match expected filename pattern for raw "
                               "channel files ('{}')"
                               .format(fname, self.inputs.in_dir,
                                       self.inputs.in_fname_re))
                continue
            paths[match.group('channel')][match.group('axis')] = op.join(
                self.inputs.in_dir, fname)

        first_echo_index = min(paths.keys())
        last_echo_index = max(paths.keys())

        for channel_i, axes in paths.items():
            # Load image real and imaginary data and remove extreme values
            img_arrays = {}
            for ax, fname in axes.items():
                img = nib.load(fname)
                img_array = img.get_fdata()
                # Replace extreme values with random value
                img_array[img_array == 2048] = 0.02 * np.random.rand()
                img_arrays[ax] = img_array

            # Calculate magnitude and phase from coil data
            cmplx = (img_arrays[self.inputs.real_label]
                     + img_arrays[self.inputs.imaginary_label] * 1j)

            # Calculate and save magnitude image
            mag_array = np.abs(cmplx)
            for echo_i in range(mag_array.shape[4]):
                mag_img = nib.Nifti1Image(mag_array, img.affine, img.header)
                mag_path = op.join(
                    mags_dir,
                    self.inputs.out_fname_str.format(channel=channel_i,
                                                    echo=echo_i))
                echo_coil_mags.append(mag_path)
                nib.save(mag_img, mag_path)

            # Save phase image
            phase_array = np.angle(cmplx)
            phase_img = nib.Nifti1Image(phase_array, img.affine,
                                        img.header)
            phase_path = op.join(
                phases_dir,
                self.inputs.out_fname_str.format(channel=channel_i,
                                                    echo=echo_i))
            echo_coil_phases.append(phase_path)
            nib.save(phase_img, phase_path)

            # Add coil data to combined coil data
            if combined_array is None:
                combined_array = deepcopy(mag_array) ** 2
                normaliser_array = deepcopy(mag_array)
            else:
                combined_array += mag_array ** 2
                normaliser_array += mag_array
        coil_mags.append(echo_coil_mags)
        coil_phases.append(echo_coil_phases)
        # Normalise combined sum of squares image, save and append
        # to list of combined echoes
        combined_array /= normaliser_array
        combined_array[np.isnan(combined_array)] = 0
        # Generate filename and append ot list of combined coil images
        combined_fname = op.join(combined_dir,
                                 'echo_{}.nii.gz'.format(echo_i))
        combined_img = nib.Nifti1Image(combined_array, img.affine,
                                        img.header)
        nib.save(combined_img, combined_fname)
        outputs['combined_images'].append(combined_fname)
        if echo_i == first_echo_index:
            outputs['first_echo'] = combined_fname
        if echo_i == last_echo_index:
            outputs['last_echo'] = combined_fname
        return outputs

    def _gen_filename(self, name):
        if name == 'combined_dir':
            fname = op.abspath(self.inputs.combined_dir
                               if isdefined(self.inputs.combined_dir)
                               else 'combined_images')
        elif name == 'magnitudes_dir':
            fname = op.abspath(self.inputs.magnitudes_dir
                               if isdefined(self.inputs.magnitudes_dir)
                               else 'magnitudes_dir')
        elif name == 'phases_dir':
            fname = op.abspath(self.inputs.phases_dir
                               if isdefined(self.inputs.phases_dir)
                               else 'phases_dir')
        else:
            assert False
        return fname


class HIPCombineChannelsInputSpec(BaseInterfaceInputSpec):

    channels_dir = Directory(exists=True, desc=(
        "Input directory containing real and imaginary images for each "
        "channel."))
    magnitude = File(genfile=True, desc="Combined magnitude image")
    phase = File(genfile=True, desc="Combined phase image")
    q = File(genfile=True, desc="Q image")


class HIPCombineChannelsOutputSpec(TraitedSpec):

    magnitude = File(exists=True, desc="Combined magnitude image")
    phase = File(exists=True, desc="Combined phase image")
    q = File(exists=True, desc="Q image")


class HIPCombineChannels(BaseInterface):
    """
    Apply Laplacian unwrapping from STI suite to each coil
    """
    input_spec = HIPCombineChannelsInputSpec
    output_spec = HIPCombineChannelsOutputSpec

    def _run_interface(self, runtime):
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        hip = None
        for fname in os.listdir(self.inputs.channels_dir):
            img = nib.load(op.join(self.inputs.channels_dir, fname))
            img_data = img.get_fdata()
            if hip is None:
                hip = np.zeros(img_data.shape[:3])
                sum_mag = np.zeros(img_data.shape[:3])
            num_echos = img_data.shape[3]
            if num_echos < 2:
                raise BananaUsageError(
                    "At least two echos required for channel magnitude {}, "
                    "found {}".format(fname, num_echos))
            cmplx_channels = np.squeeze(img_data[:, :, :, :, 0]
                                        + 1j * img_data[:, :, :, :, 1])
            phase_channels = np.angle(cmplx_channels)
            mag_channels = np.abs(cmplx_channels)
            for i, j in zip(range(0, num_echos - 1), range(1, num_echos)):
                # Get successive echos
                mag_a = mag_channels[:, :, :, i]
                mag_b = mag_channels[:, :, :, j]
                phase_a = phase_channels[:, :, :, i]
                phase_b = phase_channels[:, :, :, j]
                # Combine HIP and sum and total magnitude
                hip += mag_a * mag_b * np.exp(-1j * (phase_a - phase_b))
                sum_mag += mag_a * mag_b
        if hip is None:
            raise BananaUsageError(
                "No channels loaded from channels directory {}"
                .format(self.inputs.channels_dir))
        # Get magnitude and phase
        phase = np.angle(hip)
        mag = np.abs(hip)
        q = mag / sum_mag
        # Create NIfTI images
        phase_img = nib.Nifti1Image(phase, img.affine, img.header)
        mag_img = nib.Nifti1Image(mag, img.affine, img.header)
        q_img = nib.Nifti1Image(q, img.affine, img.header)
        # Save NIfTIs
        nib.save(phase_img, self._gen_filename('phase'))
        nib.save(mag_img, self._gen_filename('magnitude'))
        nib.save(q_img, self._gen_filename('q'))
        return outputs

    def _gen_filename(self, name):
        if name == 'magnitude':
            fname = op.abspath(self.inputs.magnitude
                               if isdefined(self.inputs.magnitude)
                               else 'magnitude')
        elif name == 'phase':
            fname = op.abspath(self.inputs.phase
                               if isdefined(self.inputs.phase) else 'phase')
        elif name == 'q':
            fname = op.abspath(self.inputs.q if isdefined(self.inputs.q)
                               else 'q')
        else:
            assert False
        return fname
