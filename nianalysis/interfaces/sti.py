from nipype.interfaces.matlab import MatlabCommand, MatlabInputSpec
from nipype.interfaces.base import TraitedSpec, traits, File
import os.path as op
import nianalysis.interfaces


MATLAB_RESOURCES = op.join(nianalysis.interfaces.RESOURCES_DIR, 'matlab')


class BaseSTIInputSpec(MatlabInputSpec):

    # Need to override value in input spec to make it non-mandatory
    script = traits.Str(
        argstr='-r \"%s;exit\"',
        desc='m-code to run',
        position=-1)


class BaseSTIOutputSpec(TraitedSpec):
    raw_output = traits.Str("Raw output of the matlab command")


class BaseSTICommand(MatlabCommand):
    """
    Base interface for STI classes
    """

    def run(self, **inputs):
        # Set the script input of the matlab spec
        self.inputs.script = self.script(**inputs)
        print(self.inputs.script)
        results = super().run(**inputs)
        stdout = results.runtime.stdout
        # Attach stdout to outputs to access matlab results
        results.outputs.raw_output = stdout
        return results

    def script(self, **inputs):
        """
        Generate script to load images, pass them to the STI function along
        with the keyword parameters
        """
        script = "set_param(0,'CharacterEncoding','UTF-8');\n"
        # Add path for nifti loader/unloader (might be unecessary
        script += "addpath(genpath('{}'));\n".format(MATLAB_RESOURCES)
        # Load all input images
        for name, _ in self.input_imgs:
            script += "{} = load_untouch_nii('{}');\n".format(
                name, getattr(self.inputs, name))
        # Create function call
        script += '[{}] = {}({}'.format(
            ', '.join(o[0] for o in self.output_imgs), self.func,
            ', '.join(tr.formatstr.format(n) for n, tr in self.input_imgs))
        if self.has_keywords:
            script += ', ' + ', '.join(
                "'{}', {}".format(kw, getattr(self.inputs, kw))
                for kw in self.keyword_args)
        script += ');\n'
        # Save all output images
        for name, trait in self.output_imgs:
            script += "{}_img = {};\n".format(name, trait.header_from)
            script += "{0}_img.img = {0};\n".format(name)
            script += "save_untouch_nii({}_img, '{}.img');\n".format(
                name, op.join(inputs['cwd'], name))
        script += 'exit;\n'
        return script

    @property
    def input_imgs(self):
        """
        Return all input traits with 'argpos' metadata, sorted by argpos
        """
        return sorted(
            ((n, t) for n, t in self.inputs.items() if t.argpos is not None),
            key=lambda x: x[1].argpos)

    @property
    def output_imgs(self):
        return sorted(
            ((n, t) for n, t in self._outputs().items()
             if t.outpos is not None),
            key=lambda x: x[1].outpos)

    @property
    def has_keywords(self):
        return any(i.keyword for i in self.inputs.traits().values())

    @property
    def keyword_args(self):
        return (n for n, i in self.inputs.items() if i.keyword)

    def _list_outputs(self):
        outputs = self._outputs().get()
        for name, _ in self.output_imgs:
            outputs[name] = self._gen_filename(name)
        return outputs


class MRPhaseUnwrapInputSpec(BaseSTIInputSpec):

    in_file = File(exists=True, mandatory=True, argpos=0, formatstr="{}",
                   desc="Input file to unwrap")
    voxelsize = traits.List([traits.Float(), traits.Float(), traits.Float()],
                             mandatory=True, keyword=True,
                             desc="Voxel size of the image")
    padsize = traits.List([12, 12, 12],
                          (traits.Int(), traits.Int(), traits.Int()),
                          usedefault=True, keyword=True,
                          desc="Padding size for each dimension")


class MRPhaseUnwrapOutputSpec(BaseSTIOutputSpec):

    out_file = File(exists=True, outpos=0, desc="Unwrapped phase image",
                    header_from='in_file')


class MRPhaseUnwrap(BaseSTICommand):

    func = 'MRPhaseUnwrap'
    input_spec = MRPhaseUnwrapInputSpec
    output_spec = MRPhaseUnwrapOutputSpec


class VSharpInputSpec(BaseSTIInputSpec):

    in_file = File(exists=True, mandatory=True, argpos=0, formatstr="{}",
                   desc="Input file to unwrap")
    mask = File(exists=True, mandatory=True, argpos=1,
                formatstr="imerode({}>0, ball(5))",
                desc="Mask file")
    voxelsize = traits.List([traits.Float(), traits.Float(), traits.Float()],
                             mandatory=True, keyword=True,
                             desc="Voxel size of the image")


class VSharpOutputSpec(BaseSTIOutputSpec):

    out_file = File(exists=True, outpos=0, desc="Unwrapped phase image",
                    header_from='in_file')
    new_mask = File(exists=True, outpos=1, desc="New mask",
                    header_from='mask')


class VSharp(BaseSTICommand):

    func = 'V_SHARP'
    input_spec = VSharpInputSpec
    output_spec = VSharpOutputSpec


