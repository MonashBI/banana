
from itertools import groupby, chain
from operator import itemgetter
from nipype.interfaces.matlab import MatlabCommand, MatlabInputSpec
from nipype.interfaces.base import TraitedSpec, traits, File
from banana.exceptions import BananaRuntimeError
import os.path as op
from banana.interfaces import MATLAB_RESOURCES
from .base import (
    BaseStiCommand, UnwrapPhase, VSharp, QSMiLSQR,
    UnwrapPhaseInputSpec, VSharpInputSpec, QSMiLSQRInputSpec,
    UnwrapPhaseOutputSpec, VSharpOutputSpec, QSMiLSQROutputSpec)


class BaseBatchStiCommand(BaseStiCommand):
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
        return super(BaseBatchStiCommand, self).run(**inputs)

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

    def _input_fname(self, name, index, **kwargs):
        inpt = super(BaseBatchStiCommand, self)._input_fname(name)
        if isinstance(inpt, list):
            inpt = inpt[index]
        return inpt

    def _output_fname(self, name, index, **kwargs):
        return name + str(index)

    def _list_outputs(self):
        outputs = self._outputs().get()
        for name, _ in self.output_imgs:
            outputs[name] = []
            for i in range(self.batch_size):
                outputs[name].append(op.abspath(self._output_fname(name, i))
                                     + '.nii')
        return outputs


class BatchUnwrapPhaseInputSpec(UnwrapPhaseInputSpec):

    in_file = traits.List(
        File(exists=True), mandatory=True, argpos=0, formatstr="{}",
        desc="Input file to unwraps")


class BatchUnwrapPhaseOutputSpec(UnwrapPhaseOutputSpec):

    out_file = traits.List(File(exists=True),
                           outpos=0, desc="Unwrapped phase images",
                           header_from='in_file')


class BatchUnwrapPhase(BaseBatchStiCommand, UnwrapPhase):

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


class BatchVSharp(BaseBatchStiCommand, VSharp):

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


class BatchQSMiLSQR(BaseBatchStiCommand, QSMiLSQR):

    input_spec = BatchQSMiLSQRInputSpec
    output_spec = BatchQSMiLSQROutputSpec
