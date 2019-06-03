import logging
from nipype.interfaces.base import (
    TraitedSpec, traits, BaseInterface, BaseInterfaceInputSpec, Directory)
logger = logging.getLogger('banana')


class AppendPathInputSpec(BaseInterfaceInputSpec):
    base_path = Directory(exists=True, mandatory=True)
    sub_paths = traits.List(
        traits.Str(), desc=())


class AppendPathOutputSpec(TraitedSpec):
    out_path = traits.Str(desc="The first echo of the combined images")


class AppendPath(BaseInterface):
    """
    Takes all REAL and IMAGINARY pairs in current directory and prepares
    them for Phase and QSM processing.

    1. Existence of pairs is checked
    2. Files are load/save cycled for formatting and rename for consistency
    3. Magnitude and Phase components are produced
    4. Coils are combined for single magnitude images per echo
    """
    input_spec = AppendPathInputSpec
    output_spec = AppendPathOutputSpec

    def _run_interface(self, runtime):
        return runtime

    def _list_outputs(self):  # @UnusedVariable
        outputs = self._outputs().get()
        return outputs
