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
        r'.*_(?P<channel>\d+)_(?P<echo>\d+)_(?P<axis>[A-Z]+)\.nii\.gz',
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

    def _list_outputs(self):  # @UnusedVariable
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
        paths = defaultdict(lambda: defaultdict(dict))
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
            paths[match.group('echo')][match.group('channel')][
                match.group('axis')] = op.join(self.inputs.in_dir, fname)

        first_echo_index = min(paths.keys())
        last_echo_index = max(paths.keys())

        for echo_i, channels in paths.items():
            # Variables to hold combined coil images
            combined_array = None
            normaliser_array = None
            echo_coil_mags = []
            echo_coil_phases = []
            for channel_i, axes in channels.items():
                # Load image real and imaginary data and remove extreme values
                img_arrays = {}
                for ax, fname in axes.items():
                    img = nib.load(fname)
                    img_array = img.get_fdata()
                    # Replace extreme values with random value
                    img_array[img_array == 2048] = 0.02 * np.random.rand()
                    img_arrays[ax] = img_array

                # Calculate magnitude and phase from coil data
                cmplx = (img_arrays[self.inputs.real_label] +
                         img_arrays[self.inputs.imaginary_label] * 1j)

                # Calculate and save magnitude image
                mag_array = np.abs(cmplx)
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

    magnitudes_dir = Directory(exists=True, desc=(
        "Input directory containing coil magnitude images."))
    phases_dir = Directory(exists=True, desc=(
        "Input directory containing coil phase images."))
    in_fname_re = traits.Str(
        'coil_(?P<channel>\d+)_(?P<echo>\d+)\.nii\.gz', usedefault=True,
        desc=("The format string used to generate the save channel filenames. "
              "Must use the 'channel' and 'echo' field names"))
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

    def _list_outputs(self):  # @UnusedVariable
        outputs = self._outputs().get()
        mag_fname = outputs['magnitude'] = self._gen_filename('magnitude')
        phase_fname = outputs['phase'] = self._gen_filename('phase')
        q_fname = outputs['q'] = self._gen_filename('q')
        mag_paths = defaultdict(dict)
        phase_paths = defaultdict(dict)
        # Compile regular expression for extracting channel, echo and
        # complex axis indices from input file names
        fname_re = re.compile(self.inputs.in_fname_re)
        for dpath, dct in ((self.inputs.magnitudes_dir, mag_paths),
                           (self.inputs.phases_dir, phase_paths)):
            for fname in os.listdir(dpath):
                match = fname_re.match(fname)
                if match is None:
                    logger.warning("Skipping '{}' file in '{}' as it doesn't "
                                   "match expected filename pattern for raw "
                                   "channel files ('{}')"
                                   .format(fname, dpath,
                                           self.inputs.in_fname_re))
                    continue
            dct[match.group('channel')][match.group('echo')] = op.join(dpath,
                                                                       fname)
        if len(mag_paths) != len(phase_paths):
            raise BananaUsageError(
                "Mismatching number of channels between magnitude and phase "
                "channels")
        hip = None
        for chann_i in mag_paths:
            if len(mag_paths[chann_i]) != 2:
                raise BananaUsageError(
                    "Expected exactly two echos for channel magnitude {}, "
                    "found {}".format(chann_i, len(mag_paths[chann_i])))
            if len(phase_paths[chann_i]) != 2:
                raise BananaUsageError(
                    "Expected exactly two echos for channel magnitude {}, "
                    "found {}".format(chann_i, len(phase_paths[chann_i])))
            mag1 = nib.load(mag_paths[chann_i][0])
            phase1 = nib.load(phase_paths[chann_i][0])
            mag2 = nib.load(mag_paths[chann_i][1])
            phase2 = nib.load(phase_paths[chann_i][1])

            # Get array data
            mag1_array = mag1.get_fdata()
            phase1_array = phase1.get_fdata()
            mag2_array = mag2.get_fdata()
            phase2_array = phase2.get_fdata()

            if hip is None:
                hip = np.zeros(mag1_array.shape)
                sum_mag = np.zeros(mag1_array.shape)
            hip += mag1_array * mag2_array * np.exp(
                -1j * (phase1_array - phase2_array))
            sum_mag += mag1_array * mag2_array
        # Get magnitude and phase
        phase = np.angle(hip)
        mag = np.abs(hip)
        q = mag / sum_mag
        # Create NIfTI images
        phase_img = nib.Nifti1Image(phase, phase1.affine, phase1.header)
        mag_img = nib.Nifti1Image(mag, mag1.affine, mag1.header)
        q_img = nib.Nifti1Image(q, mag1.affine, mag1.header)
        # Save NIfTIs
        nib.save(phase_img, phase_fname)
        nib.save(mag_img, mag_fname)
        nib.save(q_img, q_fname)
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
