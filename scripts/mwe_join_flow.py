import os
from nipype.pipeline import engine as pe
from nipype.interfaces.base import (traits, TraitedSpec, BaseInterface)


class InputSpec(TraitedSpec):

    in_file = traits.Str()


class OutputSpec(TraitedSpec):

    out_file = traits.Str()


class BInputSpec(InputSpec):

    m = traits.Int()


class DInputSpec(InputSpec):

    n = traits.Int()


class EInputSpec(TraitedSpec):

    in_files = traits.List(traits.Str())


class DummyInterface(BaseInterface):

    input_spec = InputSpec
    output_spec = OutputSpec

    def _run_interface(self, runtime):
        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = self.inputs.in_file
        return outputs


A = C = DummyInterface


class B(DummyInterface):

    input_spec = BInputSpec


class D(DummyInterface):

    input_spec = DInputSpec


class E(BaseInterface):

    input_spec = EInputSpec

    def _run_interface(self, runtime):
        return runtime


a = pe.Node(interface=A(), name="a")
a.inputs.in_file = 'a_file'
b = pe.Node(interface=B(), name="b")
b.iterables = ("m", [1, 2])
c = pe.Node(interface=C(), name="c")
d = pe.Node(interface=D(), name="d")
d.itersource = ("b", "m")
d.iterables = [("n", {1: [3, 4], 2: [5, 6]})]
my_workflow = pe.Workflow(name="my_workflow", base_dir=os.getcwd())
my_workflow.connect([(a, b, [('out_file', 'in_file')]),
                     (b, c, [('out_file', 'in_file')]),
                     (c, d, [('out_file', 'in_file')])])
e = pe.JoinNode(interface=E(), joinsource="d",
                joinfield="in_files", name="e")
my_workflow.connect(d, 'out_file',
                    e, 'in_files')

my_workflow.run()
