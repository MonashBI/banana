import logging
import os.path as op
from nipype.interfaces.base import (
    TraitedSpec, traits, BaseInterface, BaseInterfaceInputSpec, Directory,
    File, isdefined)
from banana.exceptions import BananaRuntimeError
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
    Writes a list to lines in a text file
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

class SelectRowInputSpec(BaseInterfaceInputSpec):

    in_file = traits.Either(
        File, exists=True, mandatory=True,
        desc="Input text file")

    out_file = File(desc="Extracted DW or b-zero images")

    index = traits.Int(mandatory=True, desc="The row index to select")
    column_delimiter = traits.Str(
        ' ', usedefault=True,
        desc="Delimiter used in the text file to delimit columns")
    comment_char = traits.Str(
        '#', usedefault=True,
        desc="Ignore rows starting with this character")


class SelectRowOutputSpec(TraitedSpec):
    out = traits.List(traits.Str,
                      desc="Row selected from the text file")
    out_file = File(exists=True, desc="The selected row rewritten to file")


class SelectRow(BaseInterface):
    """
    Selects a particular row of a text file and returns it as an array or
    writes it back to file
    """
    input_spec = SelectRowInputSpec
    output_spec = SelectRowOutputSpec

    def _run_interface(self, runtime):
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        with open(self.inputs.in_file) as f:
            rows = f.read().split('\n')
        rows = [r for r in rows if not r.startswith(self.inputs.comment_char)]
        try:
            row = rows[self.inputs.index]
        except IndexError:
            raise BananaRuntimeError(
                "Row index {} is out of range ({}) for {} text file"
                .format(self.inputs.index, len(rows),
                        self.inputs.in_file)) from None
        outputs['out'] = row.split(self.inputs.column_delimiter)
        if isdefined(self.inputs.out_file):
            out_file = op.abspath(self.inputs.out_file)
            with open(out_file, 'w') as f:
                f.write(row)
            outputs['out_file'] = out_file
        return outputs
