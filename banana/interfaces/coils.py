"""
A collection of interfaces used to work with raw coil channels acquired from
MRI scanners
"""
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
    channels = traits.List(File(exists=True), mandatory=True,
                           desc=("List of channels to "))
    squeeze = traits.Bool(
        False, usedefault=True,
        desc="Whether to squeeze output arrays to remove singleton dims")


class ToPolarCoordsOutputSpec(TraitedSpec):
    channel_mags = traits.List(
        File(exists=True),
        desc=("List of magnitude images for each coil for each echo"))
    channel_phases = traits.List(
        File(exists=True),
        desc=("List of phase images for each coil for each echo"))


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
        outputs = self._outputs().get()
        # Get names for output directories
        channel_mags = outputs['channel_mags'] = []
        channel_phases = outputs['channel_phases'] = []

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
    echo_times = traits.List(traits.Float, mandatory=True, desc="Echo times")


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
                hip = np.zeros(img_data.shape[:3], dtype=complex)
                sum_mag = np.zeros(img_data.shape[:3])
                r2star = np.zeros(img_data.shape[:3])
                mag = np.zeros(img_data.shape[:3])
            num_echos = img_data.shape[3]
            if num_echos < 2:
                raise BananaUsageError(
                    "At least two echos required for channel magnitude {}, "
                    "found {}".format(fname, num_echos))
            cmplx_coil = np.squeeze(img_data[:, :, :, :, 0]
                                    + 1j * img_data[:, :, :, :, 1])
            phase_coil = np.angle(cmplx_coil)
            mag_coil = np.abs(cmplx_coil)
            for i, j in zip(range(0, num_echos - 1), range(1, num_echos)):
                # Get successive echos
                mag_a = mag_coil[:, :, :, i]
                mag_b = mag_coil[:, :, :, j]
                phase_a = phase_coil[:, :, :, i]
                phase_b = phase_coil[:, :, :, j]
                # Combine HIP and sum and total magnitude
                hip += mag_a * mag_b * np.exp(-1j * (phase_a - phase_b))
                sum_mag += mag_a * mag_b
            # Calculate R2*
            sum_echo_mags = np.sum(cmplx_coil, axis=3)
            r2star += (sum_echo_mags
                       * arlo(self.inputs.echo_times, sum_echo_mags))
            mag += sum_echo_mags
        if hip is None:
            raise BananaUsageError(
                "No channels loaded from channels directory {}"
                .format(self.inputs.channels_dir))
        # Get magnitude and phase
        phase = np.angle(hip)
        mag = np.abs(hip)
        q = mag / sum_mag
        # Set filenames in output spec
        outputs['phase'] = self._gen_filename('phase')
        outputs['magnitude'] = self._gen_filename('magnitude')
        outputs['q'] = self._gen_filename('q')
        # Create NIfTI images
        phase_img = nib.Nifti1Image(phase, img.affine, img.header)
        mag_img = nib.Nifti1Image(mag, img.affine, img.header)
        q_img = nib.Nifti1Image(q, img.affine, img.header)
        # Save NIfTIs
        nib.save(phase_img, outputs['phase'])
        nib.save(mag_img, outputs['magnitude'])
        nib.save(q_img, outputs['q'])
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
        if fname.endswith('.nii'):
            fname += '.gz'
        elif not fname.endswith('nii.gz'):
            fname += '.nii.gz'
        return fname


def arlo(te, y):
    """
    Used in calculating the R2* signal

    Parameters
    ----------
    te  : list(float)
        array containing te values(in s)
    y : 4-d array
        a multi-echo data set of arbitrary dimension echo should be the
        last dimension

    Outputs
    -------
    r2 : 3-d array
        r2-map(in Hz) empty when only one echo is provided

    If you use this function please cite
    Pei M, Nguyen TD, Thimmappa ND, Salustri C, Dong F, Cooper MA, Li J,
    Prince MR, Wang Y. Algorithm for fast monoexponential fitting based
    on Auto-Regression on Linear Operations(ARLO) of data.
    Magn Reson Med. 2015 Feb
    (2): 843-50. doi: 10.1002/mrm.25137.
    Epub 2014 Mar 24. PubMed PMID: 24664497
    PubMed Central PMCID: PMC4175304.
    """
    num_echos = len(te)
    if num_echos < 2:
        return []

    if y.shape[-1] != num_echos:
        raise BananaUsageError(
            'Last dimension of y has size {}, expected {}'.format(
                y.shape[-1], num_echos))

    yy = np.zeros(y.shape[:3])
    yx = np.zeros(y.shape[:3])
    beta_yx = np.zeros(y.shape[:3])
    beta_xx = np.zeros(y.shape[:3])

    for j in range(num_echos - 2):
        alpha = ((te[j + 2] - te[j]) * (te[j + 2] - te[j]) / 2) / (te[j + 1] -
                                                                   te[j])
        tmp = (2 * te[j + 2] * te[j + 2] - te[j] * te[j + 2] - te[j] *
               te[j] + 3 * te[j] * te[j + 1] - 3 * te[j + 1] * te[j + 2]) / 6
        beta = tmp / (te[j + 2] - te[j + 1])
        gamma = tmp / (te[j + 1] - te[j])

        echo0 = y[:, :, :, j]
        echo1 = y[:, :, :, j + 1]
        echo2 = y[:, :, :, j + 2]

        # [te[j+2]-te[j]-alpha+gamma alpha-beta-gamma beta]/((te[2]-te[1])/3)
        y1 = echo0 * (te[j + 2] - te[j] - alpha + gamma) + echo1 * (alpha - beta - gamma) + echo2 * beta
        x1 = echo0 - echo2

        yy = yy + y1 * y1
        yx = yx + y1 * x1
        beta_yx = beta_yx + beta * y1 * x1
        beta_xx = beta_xx + beta * x1 * x1

    r2 = (yx + beta_xx) / (beta_yx + yy)

    # Set NaN and inf values to 0.0
    r2[~np.isfinite(r2)] = 0.0
    return r2
