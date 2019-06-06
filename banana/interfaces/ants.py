from nipype.interfaces.base import (
    TraitedSpec, traits, File, CommandLineInputSpec, CommandLine)
import os
import os.path as op
from nipype.interfaces.base import isdefined

# ants_reg_path = os.path.abspath(
#     os.path.join(os.path.dirname(__file__), 'resources', 'bash',
#                  'antsRegistrationSyN.sh'))


class AntsRegSynInputSpec(CommandLineInputSpec):

    _trans_types = ['t', 'r', 'a', 's', 'sr', 'so', 'b', 'br', 'bo']
    _precision_types = ['f', 'd']
    input_file = File(mandatory=True, desc='existing input image',
                      argstr='-m %s', exists=True)
    ref_file = File(mandatory=True, desc='existing reference image',
                    argstr='-f %s', exists=True)
    num_dimensions = traits.Int(desc='number of dimension of the input file',
                                argstr='-d %s', mandatory=True)
    out_prefix = traits.Str(
        'antsreg', argstr='-o %s', usedefault=True, mandatory=True,
        desc='A prefix that is prepended to all output files')
    transformation = traits.Enum(
        *_trans_types, argstr='-t %s',
        desc='type of transformation. t:translation, r:rigid, a:rigid+affine,'
        's:rigid+affine+deformable Syn, sr:rigid+deformable Syn, so:'
        'deformable Syn, b:rigid+affine+deformable b-spline Syn, br:'
        'rigid+deformable b-spline Syn, bo:deformable b-spline Syn')
    num_threads = traits.Int(desc='number of threads', argstr='-n %s')
    radius = traits.Float(
        desc='radius for cross correlation metric used during SyN stage'
        ' (default = 4)', argstr='-r %f')
    spline_dist = traits.Float(
        desc='spline distance for deformable B-spline SyN transform'
        ' (default = 26)', argstr='-s %f')
    ref_mask = File(
        desc='mask for the fixed image space', exists=True, argstr='-x %s')
    precision_type = traits.List(
        traits.Enum(*_precision_types), argstr='-p %s', desc='precision type '
        '(default = d). f:float, d:double')
    use_histo_match = traits.Int(desc='use histogram matching (default = 0).'
                                 '0: False, 1:True', argstr='-j %s')


class AntsRegSynOutputSpec(TraitedSpec):
    regmat = File(desc="Linear transformation matrix")
    reg_file = File(desc="Registered image")
    warp_file = File(desc="non-linear warp file")
    inv_warp = File(desc='invert of the warp file')


class AntsRegSyn(CommandLine):

    _cmd = 'antsRegistrationSyN.sh'
    input_spec = AntsRegSynInputSpec
    output_spec = AntsRegSynOutputSpec

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['regmat'] = op.abspath(self.inputs.out_prefix +
                                       '0GenericAffine.mat')
        outputs['reg_file'] = op.abspath(self.inputs.out_prefix +
                                         'Warped.nii.gz')
        if isdefined(self.inputs.transformation and
                     (self.inputs.transformation != 'r' or
                      self.inputs.transformation != 'a' or
                      self.inputs.transformation != 't')):
            outputs['warp_file'] = op.abspath(
                self.inputs.out_prefix + '1Warp.nii.gz')
            outputs['inv_warp'] = op.abspath(
                self.inputs.out_prefix + '1InverseWarp.nii.gz')

        return outputs


if __name__ == '__main__':

    interface = AntsRegSyn()
    interface.inputs.input_file = '/Users/tclose/tmp.txt'
    interface.inputs.ref_file = '/Users/tclose/tmp.txt'
    interface.inputs.num_dimensions = 3
    interface.cmdline
