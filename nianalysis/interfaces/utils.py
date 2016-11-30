import os.path
from nipype.interfaces.utility import Merge, MergeInputSpec
from nipype.interfaces.base import (
    TraitedSpec, traits, BaseInterface, CommandLineInputSpec, File,
    Directory, InputMultiPath)
from nipype.interfaces.io import FreeSurferSource


class MergeTupleOutputSpec(TraitedSpec):
    out = traits.Tuple(desc='Merged output')  # @UndefinedVariable


class MergeTuple(Merge):
    """Basic interface class to merge inputs into a single tuple

    Examples
    --------

    >>> from nipype.interfaces.utility import Merge
    >>> mi = MergeTuple(3)
    >>> mi.inputs.in1 = 1
    >>> mi.inputs.in2 = [2, 5]
    >>> mi.inputs.in3 = 3
    >>> out = mi.run()
    >>> out.outputs.out
    (1, 2, 5, 3)

    """
    input_spec = MergeInputSpec
    output_spec = MergeTupleOutputSpec

    def _list_outputs(self):
        outputs = super(MergeTuple, self)._list_outputs()
        outputs['out'] = tuple(outputs['out'])
        return outputs


class SplitSessionInputSpec(TraitedSpec):

    session = traits.Tuple(
        traits.Str(mandatory=True, desc="The subject ID"),
        traits.Str(1, mandatory=True, usedefult=True,
                   desc="The session or processed group ID"),)


class SplitSessionOutputSpec(TraitedSpec):

    subject = traits.Str(mandatory=True, desc="The subject ID")

    session = traits.Str(1, mandatory=True, usedefult=True,
                         desc="The session or processed group ID")


class SplitSession(BaseInterface):

    input_spec = SplitSessionInputSpec
    output_spec = SplitSessionOutputSpec

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['subject'] = self.inputs.session[0]
        outputs['session'] = self.inputs.session[1]
        return outputs

    def _run_interface(self, runtime):
        return runtime


class JoinPathInputSpec(TraitedSpec):
    dirname = Directory(mandatory=True, desc='directory name')
    filename = traits.Str(mandatory=True, desc='file name')


class JoinPathOutputSpec(TraitedSpec):
    path = traits.Str(mandatory=True, desc="The joined path")


class JoinPath(BaseInterface):
    """Joins a filename to a directory name"""

    input_spec = JoinPathInputSpec
    output_spec = JoinPathOutputSpec

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['path'] = os.path.join(self.inputs.dirname,
                                       self.inputs.filename)
        return outputs

    def _run_interface(self, runtime):
        return runtime


class DummyReconAllInputSpec(CommandLineInputSpec):
    subject_id = traits.Str("recon_all", argstr='-subjid %s',
                            desc='subject name', usedefault=True)
    directive = traits.Enum('all', 'autorecon1', 'autorecon2', 'autorecon2-cp',
                            'autorecon2-wm', 'autorecon2-inflate1',
                            'autorecon2-perhemi', 'autorecon3', 'localGI',
                            'qcache', argstr='-%s', desc='process directive',
                            usedefault=True, position=0)
    hemi = traits.Enum('lh', 'rh', desc='hemisphere to process',
                       argstr="-hemi %s")
    T1_files = InputMultiPath(File(exists=True), argstr='-i %s...',
                              desc='name of T1 file to process')
    T2_file = File(exists=True, argstr="-T2 %s", min_ver='5.3.0',
                   desc='Convert T2 image to orig directory')
    use_T2 = traits.Bool(argstr="-T2pial", min_ver='5.3.0',
                         desc='Use converted T2 to refine the cortical surface')
    openmp = traits.Int(argstr="-openmp %d",
                        desc="Number of processors to use in parallel")
    subjects_dir = Directory(exists=True, argstr='-sd %s', hash_files=False,
                             desc='path to subjects directory', genfile=True)
    flags = traits.Str(argstr='%s', desc='additional parameters')


class DummyReconAllIOutputSpec(FreeSurferSource.output_spec):
    subjects_dir = Directory(exists=True, desc='Freesurfer subjects directory.')
    subject_id = traits.Str(desc='Subject name for whom to retrieve data')


class DummyReconAll(BaseInterface):

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['subjects_dir'] = '/Users/tclose/Desktop/FSTest'
        outputs['subject_id'] = '1'
        return outputs
