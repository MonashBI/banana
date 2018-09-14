from nipype.interfaces.matlab import MatlabCommand, MatlabInputSpec
from nipype.interfaces.base import (
    TraitedSpec, traits, BaseInterface, BaseInterfaceInputSpec, File,
    Directory, isdefined)
from operator import attrgetter
import os.path as op
import nianalysis.interfaces

# 
# SCRIPT_TEMPLATE = (
#     "set_param(0,'CharacterEncoding','UTF-8');\n"
#     "addpath(genpath('{matlab_dir}'));\n"
#     "{{cmd}};\n"
#     "exit;\n").format()


# def matlab_cmd(cmd):
#     return MatlabCommand(script=SCRIPT_TEMPLATE.format(cmd), mfile=True)

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


class MRPhaseUnwrapOutputSpec(BaseSTIInputSpec):

    out_file = File(exists=True, outpos=0, desc="Unwrapped phase image",
                    header_from='in_file')


class MRPhaseUnwrap(BaseSTICommand):

    func = 'MRPhaseUnwrap'
    input_spec = MRPhaseUnwrapInputSpec
    output_spec = MRPhaseUnwrapOutputSpec


# class STIInputSpec(BaseInterfaceInputSpec):
#     in_dir = Directory(exists=True, mandatory=True)
#     mask_file = File(exists=True, mandatory=True)
#     echo_times = traits.List(traits.Float(), value=[20.0],
#                              desc='Echo times in ms')
#     num_channels = traits.Int(value=32, mandatory=True,
#                               desc='Number of channels')
#     out_dir = Directory(exists=True, genfile=True,
#                         "Directory to use for command outputs")
# 
# 
# class STIOutputSpec(TraitedSpec):
#     qsm = File(exists=True)
#     tissue_phase = File(exists=True)
#     tissue_mask = File(exists=True)
# 
# 
# class STI(BaseInterface):
#     input_spec = STIInputSpec
#     output_spec = STIOutputSpec
# 
#     def _run_interface(self, runtime):  # @UnusedVariable
#         mlab = matlab_cmd(
#             "QSM('{in_dir}', '{mask_file}', '{out_dir}', {echo_times}, "
#             "{num_channels})").format(
#                 in_dir=self.inputs.in_dir,
#                 mask_file=self.inputs.mask_file,
#                 out_dir=self._gen_filename('out_dir'),
#                 echo_times=self.inputs.echo_times,
#                 num_channels=self.inputs.num_channels)
#         result = mlab.run()
#         return result.runtime
# 
#     def _list_outputs(self):
#         outputs = self._outputs().get()
#         qsm_dir = op.join(self._gen_filename('out_dir'), 'QSM')
#         outputs['qsm'] = op.join(qsm_dir, 'QSM.nii.gz')
#         outputs['tissue_phase'] = op.join(qsm_dir, 'TissuePhase.nii.gz')
#         outputs['tissue_mask'] = op.join(qsm_dir, 'PhaseMask.nii.gz')
#         return outputs
# 
#     def _gen_filename(self, name):
#         if name == 'out_dir':
#             fname = (self.inputs.out_dir
#                      if isdefined(self.inputs.out_dir) else 'out_dir')
#         else:
#             assert False
#         return op.abspath(fname)
# 
# 
# class STI_SE(STI):
# 
#     def _run_interface(self, runtime):  # @UnusedVariable
#         mlab = matlab_cmd(
#             "QSM_SingleEcho('{in_dir}', '{mask_file}', '{out_dir}')").format(
#                 in_dir=self.inputs.in_dir,
#                 mask_file=self.inputs.mask_file,
#                 out_dir=self._gen_filename('out_dir'))
#         result = mlab.run()
#         return result.runtime
# 
#     def _list_outputs(self):
#         outputs = self._outputs().get()
#         qsm_dir = op.join(self._gen_filename('out_dir'), 'QSM')
#         tissue_dir = op.join(self._gen_filename('out_dir'), 'TissuePhase')
#         outputs['qsm'] = op.join(qsm_dir, 'QSM.nii.gz')
#         outputs['tissue_phase'] = op.join(tissue_dir, 'TissuePhase.nii.gz')
#         outputs['tissue_mask'] = op.join(tissue_dir, 'CoilMasks.nii.gz')
#         return outputs
