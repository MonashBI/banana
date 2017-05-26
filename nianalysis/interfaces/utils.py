import os.path
from itertools import chain
from nipype.interfaces.utility.base import Merge, MergeInputSpec
from nipype.interfaces.base import (
    DynamicTraitedSpec, BaseInterfaceInputSpec, isdefined)
from nipype.interfaces.io import IOBase, add_traits
from nipype.utils.filemanip import filename_to_list

from nipype.interfaces.base import (
    TraitedSpec, traits, BaseInterface, File,
    Directory, InputMultiPath, CommandLineInputSpec, CommandLine)
from nipype.interfaces.io import FreeSurferSource
from nianalysis.exceptions import NiAnalysisUsageError

zip_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                        'resources', 'bash', 'zip.sh'))
optiBET_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                            'resources', 'bash', 'optiBET.sh'))
set_ants_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__),
                 'resources', '.', 'set_ANTS_path.sh'))
ants_reg_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), 'resources', '.',
                 'antsRegistrationSyN.sh'))


class MergeInputSpec(DynamicTraitedSpec, BaseInterfaceInputSpec):
    axis = traits.Enum(
        'vstack', 'hstack', usedefault=True,
        desc=('direction in which to merge, hstack requires same number '
              'of elements in each input'))
    no_flatten = traits.Bool(
        False, usedefault=True,
        desc='append to outlist instead of extending in vstack mode')


class MergeOutputSpec(TraitedSpec):
    out = traits.List(desc='Merged output')


class Merge(IOBase):
    """Basic interface class to merge inputs into a single list

    Examples
    --------

    >>> from nipype.interfaces.utility import Merge
    >>> mi = Merge(3)
    >>> mi.inputs.in1 = 1
    >>> mi.inputs.in2 = [2, 5]
    >>> mi.inputs.in3 = 3
    >>> out = mi.run()
    >>> out.outputs.out
    [1, 2, 5, 3]

    """
    input_spec = MergeInputSpec
    output_spec = MergeOutputSpec

    def __init__(self, numinputs=0, **inputs):
        super(Merge, self).__init__(**inputs)
        self._numinputs = numinputs
        if numinputs > 0:
            input_names = ['in%d' % (i + 1) for i in range(numinputs)]
        elif numinputs == 0:
            input_names = ['in_lists']
        else:
            input_names = []
        add_traits(self.inputs, input_names)

    def _list_outputs(self):
        outputs = self._outputs().get()
        out = []

        if self._numinputs == 0:
            values = getattr(self.inputs, 'in_lists')
            if not isdefined(values):
                return outputs
        else:
            getval = lambda idx: getattr(self.inputs, 'in%d' % (idx + 1))  # @IgnorePep8
            values = [getval(idx) for idx in range(self._numinputs)
                      if isdefined(getval(idx))]

        if self.inputs.axis == 'vstack':
            for value in values:
                if isinstance(value, list) and not self.inputs.no_flatten:
                    out.extend(value)
                else:
                    out.append(value)
        else:
            lists = [filename_to_list(val) for val in values]
            out = [[val[i] for val in lists] for i in range(len(lists[0]))]
        if out:
            outputs['out'] = out
        return outputs


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


class OptiBETInputSpec(CommandLineInputSpec):
    input_file = File(mandatory=True, desc='existing input image',
                      argstr='-i %s', position=1, exists=True)
    use_FSL = traits.Bool(desc='use FSL for initial extraction', argstr='-f',
                          xor=['use_AFNI'])
    use_AFNI = traits.Bool(desc='use AFNI for initial extraction', argstr='-a',
                           xor=['use_FSL'])
    _xor_mask = ('mni_1mm', 'mni_2mm', 'avg')
    use_MNI_1mm = traits.Bool(
        desc='use MNI152_T1_1mm_brain_mask.nii.gz for mask', argstr='-o',
        xor=_xor_mask)
    use_MNI_2mm = traits.Bool(
        desc='use MNI152_T1_2mm_brain_mask.nii.gz for mask', argstr='-t',
        xor=_xor_mask)
    use_avg = traits.Bool(
        desc='use avg152T1_brain.nii.gz for mask', argstr='-g', xor=_xor_mask)
    debug = traits.Bool(
        desc='use debug mode (will NOT delete intermediate files)',
        argstr='-d')


class OptiBETOutputSpec(TraitedSpec):
    betted_file = File(exists=True, desc="The optiBETted image")
    betted_mask = File(exists=True, desc="The optiBETted binary mask")


class OptiBET(CommandLine):
    """Run optiBET.sh on an input image and return one brain extracted image
    and its binary mask."""

    _cmd = optiBET_path
    input_spec = OptiBETInputSpec
    output_spec = OptiBETOutputSpec
    betted_ext = '.nii.gz'

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['betted_file'] = os.path.join(
            os.getcwd(), self._gen_filename('betted_file'))
        outputs['betted_mask'] = os.path.join(
            os.getcwd(), self._gen_filename('betted_mask'))
        return outputs

    def _gen_filename(self, name):
        if name == 'betted_file':
            fid = os.path.basename(self.inputs.input_file).split('.')[0]
            fname = fid + '_optiBET_brain' + self.betted_ext
        elif name == 'betted_mask':
            fid = os.path.basename(self.inputs.input_file).split('.')[0]
            fname = fid + '_optiBET_brain_mask' + self.betted_ext
        else:
            assert False
        return fname


class SetANTsPath(CommandLine):

    _cmd = set_ants_path


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
        desc='A prefix that is prepended to all output files', argstr='-o %s',
        mandatory=True)
    transformation = traits.List(
        traits.Enum(*_trans_types), argstr='-t %s',
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
    regmat = File(exists=True, desc="Linear transformation matrix")
    reg_file = File(exists=True, desc="Registered image")
    warp_file = File(exists=True, desc="non-linear warp file")
    inv_warp = File(exist=True, desc='invert of the warp file')


class AntsRegSyn(CommandLine):

    _cmd = ants_reg_path
    input_spec = AntsRegSynInputSpec
    output_spec = AntsRegSynOutputSpec
    mat_ext = '.mat'
    img_ext = '.nii.gz'

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['regmat'] = os.path.join(os.getcwd(),
                                         self._gen_filename('regmat'))
        outputs['reg_file'] = os.path.join(os.getcwd(),
                                           self._gen_filename('reg_file'))
        if (self.inputs.transformation != 'r' or
                self.inputs.transformation != 'a' or
                self.inputs.transformation != 't'):
            outputs['warp_file'] = os.path.join(
                os.getcwd(), self._gen_filename('warp_file'))
            outputs['inv_warp'] = os.path.join(
                os.getcwd(), self._gen_filename('inv_warp'))

        return outputs

    def _gen_filename(self, name):
        if name == 'regmat':
            fid = os.path.basename(self.inputs.out_prefix)
            fname = fid + '0GenericAffine' + self.mat_ext
        elif (name == 'warp_file' and (self.inputs.transformation != 'r' or
                                       self.inputs.transformation != 'a' or
                                       self.inputs.transformation != 't')):
            fid = os.path.basename(self.inputs.out_prefix)
            fname = fid + '_1Warp' + self.img_ext
        elif (name == 'inv_warp' and (self.inputs.transformation != 'r' or
                                      self.inputs.transformation != 'a' or
                                      self.inputs.transformation != 't')):
            fid = os.path.basename(self.inputs.out_prefix)
            fname = fid + '_1InverseWarp' + self.img_ext
        elif name == 'reg_file':
            fid = os.path.basename(self.inputs.out_prefix)
            fname = fid + self.img_ext
        else:
            assert False
        return fname
