from __future__ import absolute_import
from __future__ import division
import os.path
import numpy as np
from nipype.interfaces.base import (
    TraitedSpec, BaseInterface, File, traits, isdefined)
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
import nibabel as nib
from nianalysis.exceptions import NiAnalysisError
from operator import lt, ge


class QCMetricsInputSpec(TraitedSpec):

    in_file = File(mandatory=True, desc='input qc file')

    threshold = traits.Float(
        0.25, mandatory=False, usedefault=True,
        desc=("The threshold used to determine whether a voxel is in the ROI "
              "or not, scaled to the intensity range of the image"))
    signal_radius = traits.Float(
        0.8, mandatory=False, usedefault=True,
        desc=("The radius within which to calculate the signal intensity as a "
              "fraction of the estimated radius of the phantom"))

    ghost_radii = traits.Tuple(
        traits.Float(1.2), traits.Float(1.8), mandatory=False,
        usedefault=True,
        desc=("The internal and external radii of the region within which to "
              " calculate the ghost signal as a fraction of the estimated "
              "radius of the phantom"))

    background_radius = traits.Float(
        2.25, mandatory=False, usedefault=True,
        desc=("The internal radius of the region within which to "
              " calculate the backgrouind signal as a fraction of the "
              "estimated radius of the phantom"))

    z_extent = traits.Float(
        0.8, mandatory=False, usedefault=True,
        desc=("The fraction of the z extent to include in the mask"))

    signal = File(
        genfile=True,
        desc=("The masked \"signal\" image used to calculate the SNR"))

    ghost = File(
        genfile=True,
        desc=("The masked \"ghost\" image used to calculate the ghost"
              " intensity"))

    background = File(
        genfile=True,
        desc=("The masked \"background\" image used to calculate the SNR"))


class QCMetricsOutputSpec(TraitedSpec):

    snr = traits.Float(desc='The SNR calculated from the QA phantom')

    uniformity = traits.Float(
        desc="The uniformity of the signal in the phantom")

    ghost_intensity = traits.Float(
        desc="The intensity of the ghost region of the image")

    signal = File(desc="The masked \"signal\" image used to calculate the SNR")

    ghost = File(desc=("The masked \"ghost\" image used to calculate the ghost"
                       " intensity"))

    background = File(desc=("The masked \"background\" image used to calculate"
                            " the SNR"))


class QCMetrics(BaseInterface):
    """Calculates the required metrics from the QA data"""

    input_spec = QCMetricsInputSpec
    output_spec = QCMetricsOutputSpec

    def _run_interface(self, runtime):
        return runtime

    def _list_outputs(self):
        if (self.inputs.signal_radius > self.inputs.ghost_radii[0] or
            self.inputs.ghost_radii[0] > self.inputs.ghost_radii[1] or
                self.inputs.ghost_radii[1] > self.inputs.background_radius):
            raise NiAnalysisError(
                "signal, ghost (internal), ghost (external) and background "
                "radii need to be monotonically increasing ({}, {}, {} and {} "
                "respectively".format(self.inputs.signal_radius,
                                      self.inputs.ghost_radii[0],
                                      self.inputs.ghost_radii[1],
                                      self.inputs.background_radius))
        # Load the qc data
        qc_nifti = nib.load(self.inputs.in_file)
        qc = qc_nifti.get_data()
        # Calculate the absolute threshold value
        thresh_val = qc.min() + (qc.max() - qc.min()) * self.inputs.threshold
        # Threshold the image
        thresholded = qc > thresh_val
        # Get the indices of the thresholded voxels
        x, y, z = np.nonzero(thresholded)
        # Get the limits of the thresholded image
        limits = []
        for d in (x, y, z):
            limits.append((d.min(), d.max()))
        limits = np.array(limits)
        # Get the sphere that fits between the limits
        extent = (limits[:, 1] - limits[:, 0]) // 2
        centre = limits[:, 0] + extent
        rad = np.sum(extent[:2]) // 2.0
        x, y, z = np.meshgrid(
            np.arange(qc.shape[0]),
            np.arange(qc.shape[1]),
            np.arange(qc.shape[2]), indexing='ij')
        signal_mask = self._in_volume(
            x, y, z,
            rad * self.inputs.signal_radius,
            extent[2] * self.inputs.z_extent,
            centre)
        internal_ghost_mask = self._in_volume(
            x, y, z,
            rad * self.inputs.ghost_radii[0],
            extent[2] * self.inputs.z_extent,
            centre,
            invert=True)
        external_ghost_mask = self._in_volume(
            x, y, z,
            rad * self.inputs.ghost_radii[1],
            extent[2] * self.inputs.z_extent,
            centre)
        ghost_mask = np.logical_and(internal_ghost_mask, external_ghost_mask)
        background_mask = self._in_volume(
            x, y, z,
            rad * self.inputs.background_radius,
            extent[2],
            centre,
            invert=True)
        if not background_mask.any():
            raise NiAnalysisError(
                "No voxels in background mask. Background radius {} set too "
                "high".format(self.inputs.background_radius))
#         masked_signal = qc * signal_mask
#         import matplotlib.pyplot as plt
#         for z in range(58, 196):
#             slce = qc[:, :, z]
#             slice_mask = signal_mask[:, :, z]
#             masked_slice = masked_signal[:, :, z]
#             inverted = np.logical_xor(slice_mask, slce)
#             plt.imshow(slce)
#             plt.show()
#             plt.imshow(slice_mask)
#             plt.show()
#             plt.imshow(masked_slice)
#             plt.show()
#             plt.imshow(inverted)
#             plt.show()
#             pass
        signal = qc[signal_mask]
        ghost = qc[ghost_mask]
        background = qc[background_mask]
        snr = np.sqrt(2) * np.mean(signal) / np.std(background)
        uniformity = (
            100.0 * (signal.max() - signal.min()) /
            (signal.max() + signal.min()))
        ghost_intensity = 100.0 * np.mean(signal) / np.mean(ghost)
        # Save masked images to file
        signal_fname = self._gen_filename('signal')
        ghost_fname = self._gen_filename('ghost')
        background_fname = self._gen_filename('background')
        nib.save(nib.Nifti1Image(qc * signal_mask,
                                 affine=qc_nifti.affine), signal_fname)
        nib.save(nib.Nifti1Image(qc * ghost_mask,
                                 affine=qc_nifti.affine), ghost_fname)
        nib.save(nib.Nifti1Image(qc * background_mask,
                                 affine=qc_nifti.affine),
                 background_fname)
        outputs = self._outputs().get()
        outputs['snr'] = snr
        outputs['uniformity'] = uniformity
        outputs['ghost_intensity'] = ghost_intensity
        outputs['signal'] = signal_fname
        outputs['ghost'] = ghost_fname
        outputs['background'] = background_fname
        return outputs

    @classmethod
    def _in_volume(cls, x, y, z, rad, z_extent, centre, invert=False):
        op = lt if not invert else ge
        in_vol = np.logical_and(
            op(np.sqrt((x - centre[0]) ** 2 + (y - centre[1]) ** 2), rad),
            np.abs(z - centre[2]) <= z_extent)
        return in_vol

    def plot(self, volume, slice_index=None):
        if slice_index is None:
            slice_index = volume.shape[2] // 2
        im_slice = volume[:, :, slice_index]
        plt.imshow(im_slice)

    def _gen_filename(self, name):
        if isdefined(getattr(self.inputs, name)):
            fname = getattr(self.inputs, name)
        else:
            fname = os.path.join(os.getcwd(), name + '.nii')
        return fname
