from itertools import groupby, chain
from operator import itemgetter
from nipype.interfaces.matlab import MatlabCommand, MatlabInputSpec
from nipype.interfaces.base import TraitedSpec, traits, File
import os.path as op
from nianalysis.interfaces import MATLAB_RESOURCES


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

    BATCH_INDEX = '(i)'

    def run(self, **inputs):
        # Set the script input of the matlab spec
        self.inputs.script = self.script(**inputs)
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
        script = self._set_path()
        script += self._create_param_structs()
        script += self._load_images()
        script += self._create_function_call()
        script += self._save_images(cwd=inputs['cwd'])
        script += self._exit()
        return script

    def _set_path(self):
        script = "set_param(0,'CharacterEncoding','UTF-8');\n"
        # Add path for nifti loader/unloader (might be unecessary
        script += "addpath(genpath('{}'));\n".format(MATLAB_RESOURCES)
        return script

    def _load_images(self, index=None):
        "Load all input images"
        script = ''
        for name, trait in self.input_imgs:
            suffix = self._index_suffix(index) if trait.batch else ''
            script += "{}_img{suffix} = load_untouch_nii('{}');\n".format(
                name, self._input_file(name, (index if trait.batch else None)),
                suffix=suffix)
            script += "{name}{suffix} = {name}_img{suffix}.img;\n".format(
                name=name, suffix=suffix)
        return script

    def _input_file(self, name, index=None):  # @UnusedVariable
        inpt = getattr(self.inputs, name)
        if index is not None:
            inpt = inpt[index]
        return inpt

    def _create_param_structs(self):
        "Create parameter structs"
        script = ''
        for name, struct in self.structs:
            for keyword in struct:
                script += '{}.{} = {};\n'.format(name, keyword,
                                                 getattr(self.inputs, keyword))
        return script

    def _create_function_call(self, index=None):
        "Create function call"
        # Get input arguments
        input_args = []
        for name, trait in self.input_imgs:
            if trait.batch:
                name += self._index_suffix(index) if trait.batch else ''
            if trait.format_str is not None:
                arg = getattr(self.inputs, trait.format_str).format(name)
            else:
                arg = name
            input_args.append(arg)
        # Create output arguments
        output_args = (o for o, _ in self.output_imgs)
        # Core part of function call: output array, function name + inputs
        script = '[{}] = {}({}'.format(
            ', '.join(output_args), self.func, ', '.join(input_args))
        # Append keyword arguments to function call if required
        if self.has_keywords:
            script += ', ' + ', '.join(
                "'{}', {}".format(*kw) for kw in self.keyword_args)
        script += ');\n'
        return script

    def _save_images(self, cwd, index=None):
        "Save all output images"
        script = ''
        for name, trait in self.output_imgs:
            suffix = self._index_suffix(index) if trait.batch else ''
            # Copy image to pull header from
            script += "{}_img = {}_img{suffix};\n".format(
                name, trait.header_from, suffix=suffix)
            script += "{name}_img.img = {name};\n".format(name=name)
            script += "save_untouch_nii({}_img, '{}.nii');\n".format(
                name, op.join(cwd, name))
        return script

    def _exit(self):
        return 'exit;\n'

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
        return chain(((n, getattr(self.inputs, n))
                      for n, i in self.inputs.items() if i.keyword),
                     ((n, n) for n, _ in self.structs))

    def _list_outputs(self):
        outputs = self._outputs().get()
        for name, _ in self.output_imgs:
            outputs[name] = op.abspath(name + '.nii')
        return outputs

    def _index_suffix(self, index):
        return str(index)


class BaseBatchSTICommand(BaseSTICommand):
    """
    Runs an STI command in batch mode over a range of input files instead of
    a single one
    """

    def script(self, **inputs):
        """
        Generate script to load images, pass them to the STI function along
        with the keyword parameters
        """
        script = self._set_path()
        script += self._create_param_structs()
        script += self._load_images()
        script += self._create_function_call()
        script += self._save_images(cwd=inputs['cwd'])
        script += self._exit()
        return script


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
    H = traits.List((traits.Float(), traits.Float(), traits.Float()),
                     mandatory=True, desc="Direction of the B0 field",
                     in_struct='params')


class QSMiLSQROutputSpec(BaseSTIOutputSpec):

    out_file = File(exists=True, outpos=0, desc="Unwrapped phase image",
                    header_from='in_file')


class QSMiLSQR(BaseSTICommand):

    func = 'QSM_iLSQR'
    input_spec = QSMiLSQRInputSpec
    output_spec = QSMiLSQROutputSpec
