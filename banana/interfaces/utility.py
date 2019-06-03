import logging
import os.path as op
from nipype.interfaces.base import (
    TraitedSpec, traits, BaseInterface, BaseInterfaceInputSpec, Directory,
    File)
logger = logging.getLogger('banana')


class AppendPathInputSpec(BaseInterfaceInputSpec):
    base_path = Directory(exists=True, mandatory=True)
    sub_paths = traits.List(
        traits.Str, desc=("The sub-paths to append to the "))


class AppendPathOutputSpec(TraitedSpec):
    out_path = traits.Either(
        File, Directory, desc="The first echo of the combined images")


class AppendPath(BaseInterface):
    """
    Appends a sub-path to an existing directory
    """
    input_spec = AppendPathInputSpec
    output_spec = AppendPathOutputSpec

    def _run_interface(self, runtime):
        return runtime

    def _list_outputs(self):  # @UnusedVariable
        outputs = self._outputs().get()
        outputs['out_path'] = op.join(self.inputs.base_path,
                                      *self.inputs.sub_paths)
        return outputs
