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

logger = logging.getLogger('nianalysis')


class CombineCoilsInputSpec(BaseInterfaceInputSpec):
    in_dir = Directory(exists=True, mandatory=True)
    fname_re = traits.Str(
        r'.*(?P<channel>\d+)_(?P<echo>\d+)_(?P<axis>[A-Z]+)\.nii\.gz',
        usedefault=True, desc=(
            "Regex to extract the channel, echo and complex axis from "
            "the input file name"))
    real_label = traits.Str('REAL', usedefault=True, desc=(
        "The label used to specify the real component image"))
    imaginary_label = traits.Str('IMAGINARY', usedefault=True, desc=(
        "The label used to specify the real component image"))
    combined_dir = Directory(genfile=True, desc=(
        "Output directory for coil magnitude and phase images. "
        "Files will be saved with the name "
        "'Raw_Coil_<channel>_<echo>.nii.gz'"))
    coils_dir = Directory(genfile=True, desc=(
        "Output directory for coil magnitude and phase images. "
        "Files will be saved with the name "
        "'Raw_Coil_<channel>_<echo>.nii.gz'"))


class CombineCoilsOutputSpec(TraitedSpec):
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
    coils_dir = Directory(exists=True, desc=(
        "Output directory for coil magnitude and phase images. "
        "Files will be saved with the name "
        "'Raw_Coil_<channel>_<echo>.nii.gz'"))


class CombineCoils(BaseInterface):
    """
    Takes all REAL and IMAGINARY pairs in current directory and prepares
    them for Phase and QSM processing.

    1. Existence of pairs is checked
    2. Files are load/save cycled for formatting and rename for consistency
    3. Magnitude and Phase components are produced
    4. Coils are combined for single magnitude images per echo
    """
    input_spec = CombineCoilsInputSpec
    output_spec = CombineCoilsOutputSpec

    def _run_interface(self, runtime):
        return runtime

    def _list_outputs(self):  # @UnusedVariable
        outputs = self._outputs().get()
        # Get names for output directories
        combined_dir = outputs['combined_dir'] = self._gen_filename(
            'combined_dir')
        coils_dir = outputs['coils_dir'] = self._gen_filename('coils_dir')
        # Ensure output directories exist
        os.makedirs(combined_dir, exist_ok=True)
        os.makedirs(coils_dir, exist_ok=True)
        outputs['combined_images'] = []
        coil_mags = outputs['coil_magnitudes'] = []
        coil_phases = outputs['coil_phases'] = []
        # A default dict with three levels of keys to hold the file names
        # sorted into echo, channel and complex axis
        paths = defaultdict(lambda: defaultdict(dict))
        # Compile regular expression for extracting channel, echo and
        # complex axis indices from input file names
        fname_re = re.compile(self.inputs.fname_re)
        for fname in os.listdir(self.inputs.in_dir):
            match = fname_re.match(fname)
            if match is None:
                logger.warning("Skipping '{}' file in '{}' as it doesn't "
                               "match expected filename pattern for raw "
                               "channel files ('{}')"
                               .format(fname, self.inputs.in_dir,
                                       self.inputs.fname_re))
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
                    coils_dir,
                    'Raw_Coil_{}_{}_MAGNITUDE.nii.gz'.format(channel_i,
                                                             echo_i))
                echo_coil_mags.append(mag_path)
                nib.save(mag_img, mag_path)

                # Save phase image
                phase_array = np.angle(cmplx)
                phase_img = nib.Nifti1Image(phase_array, img.affine,
                                            img.header)
                phase_path = op.join(
                    coils_dir,
                    'Raw_Coil_{}_{}_PHASE.nii.gz'.format(channel_i, echo_i))
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
        elif name == 'coils_dir':
            fname = op.abspath(self.inputs.coils_dir
                               if isdefined(self.inputs.coils_dir)
                               else 'coil_images')
        else:
            assert False
        return fname
