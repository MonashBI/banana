import os.path as op
from nipype.interfaces.matlab import MatlabCommand, MatlabInputSpec
from nipype.interfaces.base import TraitedSpec, traits, File
from banana.interfaces import MATLAB_RESOURCES


class BaseMatlabInputSpec(MatlabInputSpec):

    # Need to override value in input spec to make it non-mandatory
    script = traits.Str(
        argstr='-r \"%s;exit\"',
        desc='m-code to run',
        position=-1)


class BaseMatlabOutputSpec(TraitedSpec):

    out_file = File(exists=True, desc="Processed file")
    raw_output = traits.Str("Raw output of the matlab command")


class BaseMatlab(MatlabCommand):
    """
    Base class for MATLAB mask interfaces
    """
    output_spec = BaseMatlabOutputSpec

    def run(self, **inputs):
        self.work_dir = inputs['cwd']
        # Set the script input of the matlab spec
        self.inputs.script = (
            "set_param(0,'CharacterEncoding','UTF-8');\n"
            "addpath(genpath('{}'));\n".format(MATLAB_RESOURCES)
            + self.script(**inputs)
            + "exit;")
        results = super().run(**inputs)
        stdout = results.runtime.stdout
        # Attach stdout to outputs to access matlab results
        results.outputs.raw_output = stdout
        return results

    def script(self, **inputs):
        """
        Generate script to perform masking
        """
        raise NotImplementedError

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = self.out_file
        return outputs

    @property
    def out_file(self):
        return op.realpath(op.abspath(op.join(self.work_dir, 'out_file.nii')))
