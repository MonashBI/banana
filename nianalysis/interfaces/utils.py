import os.path
from itertools import chain
from nipype.interfaces.utility import Merge, MergeInputSpec
from nipype.interfaces.base import (
    TraitedSpec, traits, BaseInterface, File,
    Directory, InputMultiPath, CommandLineInputSpec, CommandLine)
from nipype.interfaces.io import FreeSurferSource
from nianalysis.exceptions import NiAnalysisUsageError

zip_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                        'resources', 'bash', 'zip.sh'))


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


class ChainInputSpec(TraitedSpec):

    in_lists = traits.List(traits.List(traits.Any()),
                           desc=("List of lists to chain"))


class ChainOutputSpec(TraitedSpec):

    out_list = traits.List(traits.Any(),
                           desc=("Out chained list"))


class Chain(BaseInterface):

    input_spec = ChainInputSpec
    output_spec = ChainOutputSpec

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_list'] = list(chain(*self.inputs.in_lists))
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


class ZipDirInputSpec(CommandLineInputSpec):
    dirname = Directory(mandatory=True, desc='directory name', argstr='%s',
                        position=1)
    zipped = File(genfile=True, argstr='%s', position=0,
                  desc=("The zipped zip file"))


class ZipDirOutputSpec(TraitedSpec):
    zipped = File(exists=True, desc="The zipped directory")


class ZipDir(CommandLine):
    """Creates a zip archive from a given folder"""

    _cmd = zip_path
    input_spec = ZipDirInputSpec
    output_spec = ZipDirOutputSpec
    zip_ext = '.zip'

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['zipped'] = os.path.join(os.getcwd(),
                                         self._gen_filename('zipped'))
        return outputs

    def _gen_filename(self, name):
        if name == 'zipped':
            fname = os.path.basename(self.inputs.dirname) + self.zip_ext
        else:
            assert False
        return fname


class UnzipDirInputSpec(CommandLineInputSpec):
    zipped = Directory(mandatory=True, desc='zipped file name', argstr='%s',
                       position=0)


class UnzipDirOutputSpec(TraitedSpec):
    unzipped = Directory(exists=True, desc="The unzipped directory")


class UnzipDir(CommandLine):
    """Unzips a folder that was zipped by ZipDir"""

    _cmd = 'unzip -q'
    input_spec = UnzipDirInputSpec
    output_spec = UnzipDirOutputSpec

    def _run_interface(self, *args, **kwargs):
        self.listdir_before = set(os.listdir(os.getcwd()))
        return super(UnzipDir, self)._run_interface(*args, **kwargs)

    def _list_outputs(self):
        outputs = self._outputs().get()
        new_files = set(os.listdir(os.getcwd())) - self.listdir_before
        if len(new_files) > 1:
            raise NiAnalysisUsageError(
                "Zip archives can only contain a single directory, found '{}'"
                .format("', '".join(new_files)))
        try:
            unzipped = next(iter(new_files))
        except StopIteration:
            raise NiAnalysisUsageError(
                "No files or directories found in unzipped directory")
        outputs['unzipped'] = os.path.join(os.getcwd(), unzipped)
        return outputs


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
    use_T2 = traits.Bool(
        argstr="-T2pial", min_ver='5.3.0',
        desc='Use converted T2 to refine the cortical surface')
    openmp = traits.Int(argstr="-openmp %d",
                        desc="Number of processors to use in parallel")
    subjects_dir = Directory(exists=True, argstr='-sd %s', hash_files=False,
                             desc='path to subjects directory', genfile=True)
    flags = traits.Str(argstr='%s', desc='additional parameters')


class DummyReconAllIOutputSpec(FreeSurferSource.output_spec):
    subjects_dir = Directory(exists=True,
                             desc='Freesurfer subjects directory.')
    subject_id = traits.Str(desc='Subject name for whom to retrieve data')


class DummyReconAll(BaseInterface):

    input_spec = DummyReconAllInputSpec
    output_spec = DummyReconAllIOutputSpec

    def _run_interface(self, runtime):
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['subjects_dir'] = '/Users/tclose/Desktop/FSTest'
        outputs['subject_id'] = 'recon_all'
        return outputs
