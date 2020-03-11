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


class HipCombineChannelsInputSpec(BaseInterfaceInputSpec):

    channels_dir = Directory(exists=True, desc=(
        "Input directory containing real and imaginary images for each "
        "channel."))
    magnitude = File(genfile=True, desc="Combined magnitude image")
    phase = File(genfile=True, desc="Combined phase image")
    q = File(genfile=True, desc="Q image")
    r2star = File(genfile=True, desc="R2* image")
    echo_times = traits.List(traits.Float, mandatory=True, desc="Echo times")


class HipCombineChannelsOutputSpec(TraitedSpec):

    magnitude = File(exists=True, desc="Combined magnitude image")
    phase = File(exists=True, desc="Combined phase image")
    q = File(exists=True, desc="Q image")
    r2star = File(exists=True, desc="The R2* image of the combined coils")


class HipCombineChannels(BaseInterface):
    """
    Apply Laplacian unwrapping from STI suite to each coil
    """
    input_spec = HipCombineChannelsInputSpec
    output_spec = HipCombineChannelsOutputSpec

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
            if len(self.inputs.echo_times) != num_echos:
                raise BananaUsageError(
                    "Number of echos differs from provided dataset ({}) and "
                    "echo times ({})".format(num_echos,
                                             self.inputs.echo_times))
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
            sum_echo_mags = np.sum(mag_coil, axis=3)
            r2star += sum_echo_mags * arlo(self.inputs.echo_times, mag_coil)
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
        outputs['r2star'] = self._gen_filename('r2star')
        # Create NIfTI images
        phase_img = nib.Nifti1Image(phase, img.affine, img.header)
        mag_img = nib.Nifti1Image(mag, img.affine, img.header)
        q_img = nib.Nifti1Image(q, img.affine, img.header)
        r2star_img = nib.Nifti1Image(r2star, img.affine, img.header)
        # Save NIfTIs
        nib.save(phase_img, outputs['phase'])
        nib.save(mag_img, outputs['magnitude'])
        nib.save(q_img, outputs['q'])
        nib.save(r2star_img, outputs['r2star'])
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
        elif name == 'r2star':
            fname = op.abspath(self.inputs.r2star
                               if isdefined(self.inputs.r2star) else 'r2star')
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
        y1 = (echo0 * (te[j + 2] - te[j] - alpha + gamma)
              + echo1 * (alpha - beta - gamma) + echo2 * beta)
        x1 = echo0 - echo2

        yy = yy + y1 * y1
        yx = yx + y1 * x1
        beta_yx = beta_yx + beta * y1 * x1
        beta_xx = beta_xx + beta * x1 * x1

    r2 = (yx + beta_xx) / (beta_yx + yy)

    # Set NaN and inf values to 0.0
    r2[~np.isfinite(r2)] = 0.0
    return r2


class SwiInputSpec(BaseInterfaceInputSpec):

    magnitude = File(genfile=True, desc="Magnitude image")
    tissue_phase = File(genfile=True, desc="The brain extracted phase image")
    mask = File(genfile=True, desc="The in which to use the phase image")
    out_file = File(genfile=True, desc="Path for generated SWI image")
    alpha = traits.Int(4, usedefault=True,
                       desc="The power which the phase image is raised to")


class SwiOutputSpec(TraitedSpec):

    out_file = File(exists=True, desc="SWI contrast image")


class Swi(BaseInterface):
    """
    Generate Susceptibiliy weighted from phase and magnitude
    """
    input_spec = SwiInputSpec
    output_spec = SwiOutputSpec

    def _run_interface(self, runtime):
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        mag_img = nib.load(self.inputs.magnitude)
        tissue_phase_img = nib.load(self.inputs.tissue_phase)
        mask_img = nib.load(self.inputs.mask)
        mag = mag_img.get_fdata()
        tissue_phase = tissue_phase_img.get_fdata()
        mask = np.array(mask_img.get_fdata(), dtype=bool)
        if mag.shape != tissue_phase.shape:
            raise BananaUsageError(
                "Dimensions of provided magnitude and phase images "
                "differ ({} and {})".format(mag.shape, tissue_phase.shape))
        pos_mask = (tissue_phase > 0) * mask  # Positive phase mask
        rho = np.pi - (tissue_phase / np.pi)
        rho[~pos_mask | (rho < 0)] = 1
        swi = mag * (rho ** self.inputs.alpha)
        # Set filenames in output spec
        outputs['out_file'] = self._gen_filename('out_file')
        out_file_img = nib.Nifti1Image(swi, mag_img.affine, mag_img.header)
        nib.save(out_file_img, outputs['out_file'])
        return outputs

    def _gen_filename(self, name):
        if name == 'out_file':
            fname = op.abspath(self.inputs.out_file
                               if isdefined(self.inputs.out_file) else 'swi')
        else:
            raise Exception("Unrecognised filename to generate '{}'"
                            .format(name))
        if fname.endswith('.nii'):
            fname += '.gz'
        elif not fname.endswith('nii.gz'):
            fname += '.nii.gz'
        return fname
