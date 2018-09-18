from itertools import groupby, chain
from operator import itemgetter
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
        # Create parameter structs
        for struct, keywords in self.structs:
            for name, val in keywords.items():
                script += '{}.{} = {};\n'.format(struct, name, val)
        # Create function call
        input_args = []
        for name, trait in self.input_imgs:
            if trait.format_str is not None:
                arg = getattr(self.inputs, trait.format_str).format(name)
            else:
                arg = name
            input_args.append(arg)
        output_args = (o[0] for o in self.output_imgs)
        script += '[{}] = {}({}'.format(
            ', '.join(output_args), self.func, ', '.join(input_args))
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
    def structs(self):
        key = itemgetter(0)
        for struct, keywords in groupby(sorted(
            ((t.in_struct, (n, t)) for n, t in self.inputs.items()
             if t.in_struct is not None), key=key), key=key):
            yield struct, dict(kw for _, kw in keywords)

    @property
    def has_keywords(self):
        return (any(i.keyword for i in self.inputs.traits().values()) or
                list(self.structs))

    @property
    def keyword_args(self):
        return chain((n for n, i in self.inputs.items() if i.keyword),
                     (n for n, kw in self.structs))

    def _list_outputs(self):
        outputs = self._outputs().get()
        for name, _ in self.output_imgs:
            outputs[name] = self._gen_filename(name)
        return outputs


class UnwrapPhaseInputSpec(BaseSTIInputSpec):

    in_file = File(exists=True, mandatory=True, argpos=0, formatstr="{}",
                   desc="Input file to unwrap")
    voxelsize = traits.List([traits.Float(), traits.Float(), traits.Float()],
                             mandatory=True, keyword=True,
                             desc="Voxel size of the image")
    padsize = traits.List([12, 12, 12],
                          (traits.Int(), traits.Int(), traits.Int()),
                          usedefault=True, keyword=True,
                          desc="Padding size for each dimension")


class UnwrapPhaseOutputSpec(BaseSTIOutputSpec):

    out_file = File(exists=True, outpos=0, desc="Unwrapped phase image",
                    header_from='in_file')
    dummy = File(exists=True, outpos=1, desc=(
        "Not sure, and not currently required by out workflows"))


class UnwrapPhase(BaseSTICommand):

    func = 'MRPhaseUnwrap'
    input_spec = UnwrapPhaseInputSpec
    output_spec = UnwrapPhaseOutputSpec


class VSharpInputSpec(BaseSTIInputSpec):

    in_file = File(exists=True, mandatory=True, argpos=0,
                   desc="Input file to unwrap")
    mask = File(exists=True, mandatory=True, argpos=1, desc="Mask file",
                format_str='mask_manip')
    mask_manip = traits.Str(
        desc=("A format string used to manipulate the mask before it is "
              "passed as an argument to the function"))
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


class QSMiLSQRInputSpec(BaseSTIInputSpec):

    in_file = File(exists=True, mandatory=True, argpos=0,
                   desc="Input file to unwrap")
    mask = File(exists=True, mandatory=True, argpos=1,
                desc="Input file to unwrap", format_str='mask_manip')
    mask_manip = traits.Str(
        desc=("The format string used to manipulate the mask before it is "
              "passed as an argument to the function"))
    voxelsize = traits.List([traits.Float(), traits.Float(), traits.Float()],
                             mandatory=True, in_struct='params',
                             desc="Voxel size of the image")
    padsize = traits.List([12, 12, 12],
                          (traits.Int(), traits.Int(), traits.Int()),
                          usedefault=True, in_struct='params',
                          desc="Padding size for each dimension")
    te = traits.Float(mandatory=True, desc="TE time of acquisition protocol",
                      in_struct='params')
    B0 = traits.Float(mandatory=True, desc="B0 field strength",
                      in_struct='params')
    H = traits.Tuple((traits.Int(), traits.Int(), traits.Int()),
                     mandatory=True, desc="Direction of the B0 field",
                     in_struct='params')


class QSMiLSQROutputSpec(BaseSTIOutputSpec):

    out_file = File(exists=True, outpos=0, desc="Unwrapped phase image",
                    header_from='in_file')


class QSMiLSQR(BaseSTICommand):

    func = 'QSMiLSQRUnwrap'
    input_spec = QSMiLSQRInputSpec
    output_spec = QSMiLSQROutputSpec
