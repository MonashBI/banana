from __future__ import absolute_import
from __future__ import division
import numpy as np
from nipype.interfaces.base import (
    TraitedSpec, BaseInterface, File, traits)
import nibabel as nib
import matplotlib.pyplot as plt  # @UnusedImport
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

    ghost_radius = traits.Tuple(
        (1.2, 1.8), traits.Float(), mandatory=False, usedefault=True,
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
        if (self.inputs.signal_radius > self.inputs.ghost_radius[0] or
            self.inputs.ghost_radius[0] > self.inputs.ghost_radius[1] or
                self.inputs.ghost_radius[1] > self.inputs.background_radius):
            raise NiAnalysisError(
                "signal, ghost (internal), ghost (external) and background "
                "radii need to be monotonically increasing ({}, {}, {} and {} "
                "respectively".format(self.inputs.signal_radius,
                                      self.inputs.ghost_radius[0],
                                      self.inputs.ghost_radius[1],
                                      self.inputs.background_radius))
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
            x, y, z,
            rad * self.inputs.signal_radius,
            extent[2] * self.inputs.z_extent,
            centre)
        internal_ghost_mask = self._in_volume(
            x, y, z,
            rad * self.inputs.ghost_radius[0],
            extent[2] * self.inputs.z_extent,
            centre,
            invert=True)
        external_ghost_mask = self._in_volume(
            x, y, z,
            rad * self.inputs.ghost_radius[1],
            extent[2] * self.inputs.z_extent,
            centre)
        ghost_mask = np.logical_and(internal_ghost_mask, external_ghost_mask)
        background_mask = self._in_volume(
            x, y, z,
            rad * self.inputs.background_radius,
            extent[2] * self.inputs.z_extent,
            centre,
            invert=True)
        if not background_mask.any():
            raise NiAnalysisError(
                "No voxels in background mask. Background radius {} set too "
                "high".format(self.inputs.background_radius))
        signal = qc[signal_mask]
        ghost = qc[ghost_mask]
        background = qc[background_mask]
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
        op = lt if not invert else ge
        in_vol = np.logical_and(
            op(np.sqrt((x - centre[0]) ** 2 + (y - centre[1]) ** 2), rad),
            np.abs(z - centre[2]) <= z_extent)
        return in_vol
