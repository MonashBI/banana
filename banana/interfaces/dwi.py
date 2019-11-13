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
