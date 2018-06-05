
from nipype.interfaces.base import (
    isdefined, CommandLineInputSpec, TraitedSpec, CommandLine, File,
    traits)
import os
from arcana.utils import split_extension


class ANTs2FSLMatrixConversionInputSpec(CommandLineInputSpec):

    itk_file = File(exists=True, argstr="-itk %s", position=1)
    reference_file = File(exists=True, argstr="-ref %s", position=2)
    source_file = File(exists=True, argstr='-src %s', position=3)
    ras2fsl = traits.Bool(argstr='-ras2fsl', position=4)
    fsl_matrix = File(genfile=True, argstr='-o %s', position=5)


class ANTs2FSLMatrixConversionOutputSpec(TraitedSpec):

    fsl_matrix = File(exists=True)


class ANTs2FSLMatrixConversion(CommandLine):

    _cmd = 'c3d_affine_tool'
    input_spec = ANTs2FSLMatrixConversionInputSpec
    output_spec = ANTs2FSLMatrixConversionOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['fsl_matrix'] = self._gen_outfilename()
        return outputs

    def _gen_filename(self, name):
        if name == 'fsl_matrix':
            gen_name = self._gen_outfilename()
        else:
            assert False
        return gen_name

    def _gen_outfilename(self):
        if isdefined(self.inputs.fsl_matrix):
            out_name = self.inputs.fsl_matrix
        else:
            base, _ = split_extension(
                os.path.basename(self.inputs.itk_file))
            out_name = os.path.join(os.getcwd(),
                                    "{}_fsl.mat".format(base))
        return out_name
