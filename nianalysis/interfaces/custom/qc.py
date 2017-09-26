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
    buffer = traits.Tuple(
        traits.Float(), (0.2, 0.2), mandatory=False, usedefault=True,
        desc=("The internal and external buffer (as a fraction) subtracted "
              "and added to the radius when determining the phantom and "
              "non-phantom masks respectively"))


class QCMetricsOutputSpec(TraitedSpec):

    snr = traits.Float(desc='The SNR calculated from the QC phantom')


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
        phantom_mask = self._in_volume(x, y, z,
                                       rad * (1 - self.inputs.buffer[0]),
                                       extent[2],
                                       centre)
        non_phantom_mask = self._in_volume(x, y, z,
                                           rad * (1 + self.inputs.buffer[1]),
                                           extent[2],
                                           centre, invert=True)
        phantom = qc[phantom_mask]
        non_phantom = qc[non_phantom_mask]
        plt_phantom = np.array(qc)
        plt_non_phantom = np.array(qc)
        plt_phantom[np.logical_not(phantom_mask)] = 0.0
        plt_non_phantom[np.logical_not(non_phantom_mask)] = 0.0
        plt.imshow(plt_phantom[:, :, 100])
        plt.imshow(plt_non_phantom[:, :, 100])
        snr = np.sqrt(2) * np.mean(phantom) / np.std(non_phantom)
        outputs = self._outputs().get()
        outputs['snr'] = snr
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
