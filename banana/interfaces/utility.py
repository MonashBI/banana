import logging
import os.path as op
from nipype.interfaces.base import (
    TraitedSpec, traits, BaseInterface, BaseInterfaceInputSpec, Directory,
    File)
from nipype.interfaces.utility import Function
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

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_path'] = op.join(self.inputs.base_path,
                                      *self.inputs.sub_paths)
        return outputs


def list_to_file(in_list, file_name):
    with open(file_name, 'w') as f:
        f.write('\n'.join(str(l) for l in in_list))
        

class ListToFileInputSpec(BaseInterfaceInputSpec):
    in_list = traits.List(
        traits.Either(traits.Int, traits.Float, traits.Str), mandatory=True,
        desc="List to write to a text file")
    out_fname = File('out.txt', usedefault=True,
                     desc=("The sub-paths to append to the "))


class ListToFileOutputSpec(TraitedSpec):
    out = File(exists=True, desc="The first echo of the combined images")


class ListToFile(BaseInterface):
    """
    Appends a sub-path to an existing directory
    """
    input_spec = ListToFileInputSpec
    output_spec = ListToFileOutputSpec

    def _run_interface(self, runtime):
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        with open(self.inputs.out_fname, 'w') as f:
            f.write('\n'.join(str(l) for l in self.inputs.in_list))
        outputs['out'] = op.abspath(self.inputs.out_fname)
        return outputs
