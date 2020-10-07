from __future__ import absolute_import
import os.path as op
import numpy as np
from nipype.interfaces.base import (
    TraitedSpec, BaseInterface, File, isdefined, traits)


class TransformGradientsInputSpec(TraitedSpec):
    gradients = File(exists=True, mandatory=True,
                     desc='input gradients to transform in FSL bvec format')
    transform = File(desc='The affine transform (output from FLIRT)')
    transformed = traits.Str(desc='the name for the transformed file')


class TransformGradientsOutputSpec(TraitedSpec):
    transformed = File(exists=True, desc='the transformed gradients')


class TransformGradients(BaseInterface):
    """
    Applies a coregistration transform matrix (in FLIRT format) to a FSL bvec
    file
    """

    input_spec = TransformGradientsInputSpec
    output_spec = TransformGradientsOutputSpec

    def _run_interface(self, runtime):
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        rotation = np.loadtxt(self.inputs.transform)[:3, :3]
        gradients = np.loadtxt(self.inputs.gradients)
        transformed = rotation.dot(gradients)
        np.savetxt(self.transformed_path, transformed)
        outputs['transformed'] = self.transformed_path
        return outputs

    @property
    def transformed_path(self):
        if isdefined(self.inputs.transformed):
            dpath = self.inputs.transformed
        else:
            dpath = op.abspath('transformed')
        return dpath


class SelectShellInputSpec(TraitedSpec):

    bvals = File(exists=True, mandatory=True,
                 desc=("The b-values to select from"))
    target = traits.Float(mandatory=True,
                          desc="The b-value to select from list")
    tol = traits.Float(
        5.0, desc="The tolerance between the target and actual b-values")


class SelectShellOutputSpec(TraitedSpec):

    indices = traits.Str(
        desc="Indices of matching b-values in a comma-delimited list")


class SelectShell(BaseInterface):
    """
    Selects indices from FSL bval that match the target b-value
    """

    input_spec = SelectShellInputSpec
    output_spec = SelectShellOutputSpec

    def _run_interface(self, runtime):

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        with open(self.inputs.bvals) as f:
            bvals = [float(b) for b in f.read().split()]
        outputs['indices'] = ','.join(
            str(i) for i, b in enumerate(bvals)
            if (abs(b - self.inputs.target) < self.inputs.tol
                or b < self.inputs.tol))
        return outputs
