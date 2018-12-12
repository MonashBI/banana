from itertools import groupby, chain
from operator import itemgetter
from nipype.interfaces.matlab import MatlabCommand, MatlabInputSpec
from nipype.interfaces.base import TraitedSpec, traits, File
from banana.exceptions import BananaRuntimeError
import os.path as op
from banana.interfaces import MATLAB_RESOURCES


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
        self.inputs.script = self.script(cwd=inputs['cwd'])
        results = super().run(**inputs)
        stdout = results.runtime.stdout
        # Attach stdout to outputs to access matlab results
        results.outputs.raw_output = stdout
        return results

    def script(self, cwd, **kwargs):
        """
        Generate script to load images, pass them to the STI function along
        with the keyword parameters
        """
        script = self._set_path()
        script += self._create_param_structs()
        script += self._process_image(cwd, **kwargs)
        script += self._exit()
        return script

    def _process_image(self, cwd, **kwargs):
        script = self._load_images(**kwargs)
        script += self._create_function_call()
        script += self._save_images(cwd=cwd, **kwargs)
        return script

    def _set_path(self):
        script = "set_param(0,'CharacterEncoding','UTF-8');\n"
        # Add path for nifti loader/unloader (might be unecessary
        script += "addpath(genpath('{}'));\n".format(MATLAB_RESOURCES)
        return script

    def _load_images(self, **kwargs):
        "Load all input images"
        script = ''
        for name, _ in self.input_imgs:
            script += "{}_img = load_untouch_nii('{}');\n".format(
                name, self._input_fname(name, **kwargs))
            script += "{name} = {name}_img.img;\n".format(
                name=name)
        return script

    def _input_fname(self, name, **kwargs):  # @UnusedVariable
        return getattr(self.inputs, name)

    def _output_fname(self, name, **kwargs):  # @UnusedVariable
        return name

    def _create_param_structs(self):
        "Create parameter structs"
        script = ''
        for name, struct in self.structs:
            for keyword in struct:
                script += '{}.{} = {};\n'.format(name, keyword,
                                                 getattr(self.inputs, keyword))
        return script

    def _create_function_call(self):
        "Create function call"
        # Get input arguments
        input_args = []
        for name, trait in self.input_imgs:
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

    def _save_images(self, cwd, **kwargs):
        "Save all output images"
        script = ''
        for name, trait in self.output_imgs:
            # Copy image to pull header from
            script += "{}_img = {}_img;\n".format(
                name, trait.header_from)
            script += "{name}_img.img = {name};\n".format(name=name)
            script += "save_untouch_nii({}_img, '{}.nii');\n".format(
                name, op.join(cwd, self._output_fname(name, **kwargs)))
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
            outputs[name] = op.abspath(self._output_fname(name)) + '.nii.gz'
        return outputs

    def _index_suffix(self, index):
        return str(index)


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


class BaseBatchSTICommand(BaseSTICommand):
    """
    Runs an STI command in batch mode over a range of input files instead of
    a single one.

    Although a MapNode could be used instead, it would incur a penalty of
    opening and closing MATLAB for each iteration of the batch.
    """

    def run(self, **inputs):
        # Determine batch size
        self.batch_size = None
        for img_name, _ in self.input_imgs:
            inpt = getattr(self.inputs, img_name)
            if isinstance(inpt, list):
                if self.batch_size is None:
                    self.batch_size = len(inpt)
                elif self.batch_size != len(inpt):
                    raise BananaRuntimeError(
                        "Inconsistent length of batch input lists ({} and {})"
                        .format(len(inpt), self.batch_size))
        return super(BaseBatchSTICommand, self).run(**inputs)

    def script(self, cwd, **kwargs):
        """
        Generate script to load images, pass them to the STI function along
        with the keyword parameters
        """
        script = self._set_path()
        script += self._create_param_structs()
        for i in range(self.batch_size):
            script += self._process_image(cwd, index=i, **kwargs)
        script += self._exit()
        return script

    def _input_fname(self, name, index, **kwargs):  # @UnusedVariable
        inpt = super(BaseBatchSTICommand, self)._input_fname(name)
        if isinstance(inpt, list):
            inpt = inpt[index]
        return inpt

    def _output_fname(self, name, index, **kwargs):  # @UnusedVariable
        return name + str(index)

    def _list_outputs(self):
        outputs = self._outputs().get()
        for name, _ in self.output_imgs:
            outputs[name] = []
            for i in range(self.batch_size):
                outputs[name].append(op.abspath(self._output_fname(name, i)) +
                                     '.nii')
        return outputs


class BatchUnwrapPhaseInputSpec(UnwrapPhaseInputSpec):

    in_file = traits.List(
        File(exists=True), mandatory=True, argpos=0, formatstr="{}",
        desc="Input file to unwraps")


class BatchUnwrapPhaseOutputSpec(UnwrapPhaseOutputSpec):

    out_file = traits.List(File(exists=True),
                           outpos=0, desc="Unwrapped phase images",
                           header_from='in_file')


class BatchUnwrapPhase(BaseBatchSTICommand, UnwrapPhase):

    input_spec = BatchUnwrapPhaseInputSpec
    output_spec = BatchUnwrapPhaseOutputSpec


class BatchVSharpInputSpec(VSharpInputSpec):

    in_file = traits.List(File(exists=True), mandatory=True, argpos=0,
                          desc="Input files to unwrap")
    mask = traits.List(File(exists=True), mandatory=True, argpos=1,
                       desc="Mask files", format_str='mask_manip')


class BatchVSharpOutputSpec(VSharpOutputSpec):

    out_file = traits.List(
        File(exists=True), outpos=0, desc="Unwrapped phase images",
        header_from='in_file')
    new_mask = traits.List(
        File(exists=True), outpos=1, desc="New masks", header_from='mask')


class BatchVSharp(BaseBatchSTICommand, VSharp):

    input_spec = BatchVSharpInputSpec
    output_spec = BatchVSharpOutputSpec


class BatchQSMiLSQRInputSpec(QSMiLSQRInputSpec):

    in_file = traits.List(
        File(exists=True), mandatory=True, argpos=0,
        desc="Input files to unwrap")
    mask = traits.List(
        File(exists=True), mandatory=True, argpos=1,
        desc="Input files to unwrap", format_str='mask_manip')


class BatchQSMiLSQROutputSpec(QSMiLSQROutputSpec):

    out_file = traits.List(
        File(exists=True), outpos=0,
        desc="Unwrapped phase images", header_from='in_file')


class BatchQSMiLSQR(BaseBatchSTICommand, QSMiLSQR):

    input_spec = BatchQSMiLSQRInputSpec
    output_spec = BatchQSMiLSQROutputSpec
