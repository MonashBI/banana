from nipype.interfaces.base import TraitedSpec, traits, BaseInterface
from nipype.interfaces.utility import Merge, MergeInputSpec


class MergeTupleOutputSpec(TraitedSpec):
    out = traits.Tuple(desc='Merged output')  # @UndefinedVariable


class MergeTuple(Merge):
    """Basic interface class to merge inputs into a single tuple

    Examples
    --------

    >>> from nipype.interfaces.utility import Merge
    >>> mi = MergeTuple(3)
    >>> mi.inputs.in1 = 1
    >>> mi.inputs.in2 = [2, 5]
    >>> mi.inputs.in3 = 3
    >>> out = mi.run()
    >>> out.outputs.out
    (1, 2, 5, 3)

    """
    input_spec = MergeInputSpec
    output_spec = MergeTupleOutputSpec

    def _list_outputs(self):
        outputs = super(MergeTuple, self)._list_outputs()
        outputs['out'] = tuple(outputs['out'])
        return outputs


class SplitSessionInputSpec(TraitedSpec):

    session = traits.Tuple(
        traits.Str(mandatory=True, desc="The subject ID"),
        traits.Str(1, mandatory=True, usedefult=True,
                   desc="The session or processed group ID"),)


class SplitSessionOutpuSpec(TraitedSpec):

    subject = traits.Str(mandatory=True, desc="The subject ID")

    study = traits.Str(1, mandatory=True, usedefult=True,
                       desc="The session or processed group ID")


class SplitSession(BaseInterface):

    input_spec = SplitSessionInputSpec
    output_spec = SplitSessionOutpuSpec

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['subject'] = self.inputs.session[0]
        outputs['study'] = self.inputs.session[1]
        return outputs

    def _run_interface(self, runtime):
        return runtime
