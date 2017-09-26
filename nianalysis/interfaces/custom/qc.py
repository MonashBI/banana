from __future__ import absolute_import
from __future__ import division
import numpy as np
from nipype.interfaces.base import (
    TraitedSpec, BaseInterface, File, traits)
import nibabel as nib
import matplotlib.pyplot as plt  # @UnusedImport


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

    ghost_radius = traits.Tuple(
        traits.Float(), (1.2, 1.6), mandatory=False, usedefault=True,
        desc=("The internal and external radii of the region within which to "
              " calculate the ghost signal as a fraction of the estimated "
              "radius of the phantom"))

    background_radius = traits.Float(
        1.6, mandatory=False, usedefault=True,
        desc=("The internal radius of the region within which to "
              " calculate the backgrouind signal as a fraction of the "
              "estimated radius of the phantom"))


class QCMetricsOutputSpec(TraitedSpec):

    snr = traits.Float(desc='The SNR calculated from the QC phantom')

    uniformity = traits.Float(
        desc="The uniformity of the signal in the phantom")

    ghost_intensity = traits.Float(
        desc="The intensity of the ghost region of the image")


class QCMetrics(BaseInterface):
    """Calculates the required metrics from the QC data"""

    input_spec = QCMetricsInputSpec
    output_spec = QCMetricsOutputSpec

    def _run_interface(self, runtime):
        return runtime

    def _list_outputs(self):
        # Load the qc data
        qc = nib.load(self.inputs.in_file).get_data()
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
        rad = np.sum(extent[:2]) / 2.0
        x, y, z = np.meshgrid(
            np.arange(qc.shape[0]),
            np.arange(qc.shape[1]),
            np.arange(qc.shape[2]))
        signal_mask = self._in_volume(
            x, y, z, rad * self.inputs.signal_radius, extent[2], centre)
        internal_ghost_mask = self._in_volume(
            x, y, z, rad * self.inputs.ghost_radius[0], extent[2], centre,
            invert=True)
        external_ghost_mask = self._in_volume(
            x, y, z, rad * self.inputs.ghost_radius[1], extent[2], centre)
        ghost_mask = np.logical_and(internal_ghost_mask, external_ghost_mask)
        background_mask = self._in_volume(
            x, y, z, rad * self.inputs.background_radius, extent[2], centre)
        signal = qc[signal_mask]
        ghost = qc[ghost_mask]
        background = qc[background_mask]
        plt_signal = np.array(qc)
        plt_ghost = np.array(qc)
        plt_background = np.array(qc)
        plt_signal[np.logical_not(signal_mask)] = 0.0
        plt_ghost[np.logical_not(ghost_mask)] = 0.0
        plt_ghost[np.logical_not(background_mask)] = 0.0
        plt.imshow(plt_signal[:, :, 100])
        plt.imshow(plt_ghost[:, :, 100])
        plt.imshow(plt_background[:, :, 100])
        snr = np.sqrt(2) * np.mean(signal) / np.std(background)
        uniformity = (
            100.0 * (signal.max() - signal.min()) /
            (signal.max() + signal.min()))
        ghost_intensity = 100.0 * np.mean(signal) / np.mean(ghost)
        outputs = self._outputs().get()
        outputs['snr'] = snr
        outputs['uniformity'] = uniformity
        outputs['ghost_intensity'] = ghost_intensity
        return outputs

    @classmethod
    def _in_volume(cls, x, y, z, rad, z_extent, centre, invert=False):
        in_vol = np.logical_and(
            np.sqrt((x - centre[0]) ** 2 +
                    (y - centre[1]) ** 2) < rad,
            np.abs(z - centre[2]) < z_extent)
        if invert:
            in_vol = np.logical_not(in_vol)
        return in_vol
