import os.path
import warnings
from string import Template
from nibabel import load
from nipype.interfaces.base import (
    File, traits, TraitedSpec, BaseInterface, BaseInterfaceInputSpec,
    Directory)
from glob import glob
from nipype.interfaces.fsl.base import (FSLCommand, FSLCommandInputSpec)
import logging
from nipype.interfaces.traits_extension import isdefined


warn = warnings.warn
warnings.filterwarnings('always', category=UserWarning)

feat_template_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "resources", 'temp.fsf')
optiBET_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                            'resources', 'bash', 'optiBET.sh'))

logger = logging.getLogger('NiAnalysis')


class MelodicL1FSFInputSpec(BaseInterfaceInputSpec):

    output_dir = traits.String(desc="Output directory", mandatory=True)

    tr = traits.Float(desc="TR", mandatory=True)
    brain_thresh = traits.Float(
        desc="Brain/Background threshold %", mandatory=True)
    dwell_time = traits.Float(desc="EPI dwell time", mandatory=True)
    te = traits.Float(desc="TE", mandatory=True)
    unwarp_dir = traits.Str(desc="Unwrap direction", mandatory=True)
    sfwhm = traits.Float(desc="Smoothing FWHM", mandatory=True)
    fmri = File(exists=True, desc="4D functional data", mandatory=True)
    fmri_ref = File(
        exists=True, desc="reference functional data", mandatory=True)
    fmap = File(exists=True, desc="fieldmap file", mandatory=True)
    fmap_mag = File(
        exists=True, desc="fieldmap magnitude file", mandatory=True)
    structural = File(exist=True, desc="Structural image", mandatory=True)
    high_pass = traits.Float(desc="high pass cutoff (s)", mandatory=True)


class MelodicL1FSFOutputSpec(TraitedSpec):

    fsf_file = File(exists=True)


class MelodicL1FSF(BaseInterface):

    def get_vols(self, nifti):
        img = load(nifti)
        if len(img.shape) < 4:
            return 1
        else:
            return img.shape[3]

    input_spec = MelodicL1FSFInputSpec
    output_spec = MelodicL1FSFOutputSpec

    def _run_interface(self, runtime):
        template_file = open(feat_template_path)
        template = Template(template_file.read())
        template_file.close()
#        inputs = self.input_spec().get()
        d = {}
        print self.inputs
        d['outputdir'] = self.inputs.output_dir
        d['tr'] = self.inputs.tr
        d['volumes'] = self.get_vols(self.inputs.fmri)
        d['del_volume'] = 0
        d['brain_thresh'] = self.inputs.brain_thresh
        d['epi_dwelltime'] = self.inputs.dwell_time
        d['epi_te'] = self.inputs.te
        d['unwarp_dir'] = self.inputs.unwarp_dir
        d['smoothing_fwhm'] = self.inputs.sfwhm
        d['highpass_cutoff'] = self.inputs.high_pass
        d['fmri_file'] = self.inputs.fmri
        d['fmri_reference'] = self.inputs.fmri_ref
        d['fieldmap'] = self.inputs.fmap
        d['fieldmap_mag'] = self.inputs.fmap_mag
        d['structural'] = self.inputs.structural
        d['fsl_dir'] = os.environ['FSLDIR']
        out = template.substitute(d)

        tt = open('melodic.fsf', 'w')
        tt.write(out)
        tt.close()
        return runtime

    def _list_outputs(self):

        outputs = self.output_spec().get()
        outputs['fsf_file'] = os.path.abspath('./melodic.fsf')
        return outputs


class FSLFIXInputSpec(FSLCommandInputSpec):
    _xor_options = ('classification', 'regression', 'all')
    _xor_input_files = ('feat_dir', 'labelled_component')
    feat_dir = Directory(
        exists=True, argstr="%s", position=1, xor=_xor_input_files,
        desc="Input melodic preprocessed directory")
    train_data = File(exists=True, argstr="%s", position=2,
                      desc="Training file")
    regression = traits.Bool(desc='Regress previously classified components.',
                             position=0, argstr="-a", xor=_xor_options)
    classification = traits.Bool(
        desc='Components classification without regression.', position=0,
        argstr="-c", xor=_xor_options)
    all = traits.Bool(
        desc='Components classification and regression.', position=0,
        argstr="-f", xor=_xor_options)
    component_threshold = traits.Int(
        argstr="%d", mandatory=True, position=3,
        desc="threshold for the number of components")
    labelled_component = File(
        exists=True, argstr="%s", position=1, xor=_xor_input_files,
        desc=("Text file with classified components. This file is mandatory if"
              "you choose regression only."))
    motion_reg = traits.Bool(argstr='-m', desc="motion parameters regression")
    highpass = traits.Float(
        argstr='-h %f', desc='apply highpass of the motion '
        'confound with <highpass> being full-width (2*sigma) in seconds.')


class FSLFIXOutputSpec(TraitedSpec):
    output = File(exists=True, desc="cleaned output")
    label_file = File(exists=True, desc="labelled components")


class FSLFIX(FSLCommand):

    _cmd = 'fix'
    input_spec = FSLFIXInputSpec
    output_spec = FSLFIXOutputSpec
    text_ext = '.txt'

    def _list_outputs(self):
        outputs = self.output_spec().get()
        print self.inputs.feat_dir+'./filtered_func_data_clean.nii*'
        if self.inputs.all:
            outputs['output'] = self._gen_filename('out_file')
            outputs['label_file'] = self._gen_filename('label_file')
        elif self.inputs.classification:
            outputs['label_file'] = self._gen_filename('label_file')
        elif self.inputs.regression:
            outputs['output'] = self._gen_filename('out_file')
        else:
            outputs['output'] = self._gen_filename('out_file')
            outputs['label_file'] = self._gen_filename('label_file')
        return outputs

    def _gen_filename(self, name):
        if isdefined(self.inputs.feat_dir):
            cwd = self.inputs.feat_dir
        elif isdefined(self.inputs.labelled_component):
            cwd = '/'.join(self.inputs.labelled_component.split('/')[:-1])

        if name == 'out_file':
            fname = cwd+glob('/filtered_func_data_clean.nii*')[0]
        elif name == 'label_file':
            fid = os.path.basename(self.inputs.train_data).split('.RData')[0]
            thr = str(self.inputs.component_threshold)
            fname = cwd+'/fix4melview_'+fid+'_thr'+thr+self.text_ext
        else:
            assert False
        return fname


class FSLFixTrainingInputSpec(FSLCommandInputSpec):
    training = traits.Bool(mandatory=True, argstr="-t", position=1)
    list_dir = traits.List(mandatory=True, argstr="%s", position=-1,
                           desc="Input feat preprocessed directory")
    outname = traits.Str(mandatory=True, argstr="%s", position=2,
                         desc="output name")


class FSLFixTrainingOutputSpec(TraitedSpec):
    training_set = File(exists=True, desc="training set")


class FSLFixTraining(FSLCommand):

    _cmd = 'fix'
    input_spec = FSLFixTrainingInputSpec
    output_spec = FSLFixTrainingOutputSpec
    ext = '.RData'

    def _list_outputs(self):
        outputs = self.output_spec().get()
        # print self.inputs.feat_dir+'./filtered_func_data_clean.nii*'
        outputs['training_set'] = os.path.join(
            os.getcwd(), self._gen_filename('train_file'))
        return outputs

    def _gen_filename(self, name):
        if name == 'train_file':
            fid = os.path.basename(self.inputs.outname)
            fname = fid + self.ext
        else:
            assert False
        return fname


class CheckLabelFileInputSpec(BaseInterfaceInputSpec):
    in_list = traits.List(desc='melodic directory', mandatory=True)


class CheckLabelFileOutputSpec(TraitedSpec):
    out_list = traits.List(desc="List of melodic dirs that contain "
                                "label file")


class CheckLabelFile(BaseInterface):
    input_spec = CheckLabelFileInputSpec
    output_spec = CheckLabelFileOutputSpec

    def _run_interface(self, runtime):
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        out = []

        for s in self.inputs.in_list:
            if 'hand_labels_noise.txt' in os.listdir(s):
                with open(s+'/hand_labels_noise.txt', 'r') as f:
                    for line in f:
                        el = [x for x in
                              line.strip().strip('[').strip(']').split(',')]
                    f.close()
                    ic = sorted(glob(s+'/report/f*.txt'))
                    if [x for x in el if int(x) > len(ic)]:
                        logger.warning('Subject {} has wrong number of '
                                       'components in the '
                                       'hand_labels_noise.txt file. It will '
                                       'not be used for the FIX training.'
                                       .format(s))
                    else:
                        out.append(s)

        outputs["out_list"] = out
        return outputs


class FSLSlicesInputSpec(FSLCommandInputSpec):
    im1 = File(mandatory=True, position=0, argstr="%s", desc="First image")
    im2 = File(mandatory=True, position=1, argstr="%s", desc="Second image")
    outname = traits.Str(mandatory=True, argstr="-o %s.gif", position=-1,
                         desc="output name")


class FSLSlicesOutputSpec(TraitedSpec):
    report = File(exists=True, desc=".gif file with im2 overlaid to im1")


class FSLSlices(FSLCommand):

    _cmd = 'slices '
    input_spec = FSLSlicesInputSpec
    output_spec = FSLSlicesOutputSpec
    ext = '.gif'

    def _list_outputs(self):
        outputs = self.output_spec().get()
        # print self.inputs.feat_dir+'./filtered_func_data_clean.nii*'
        outputs['report'] = os.path.join(
            os.getcwd(), self._gen_filename('report'))
        return outputs

    def _gen_filename(self, name):
        if name == 'report':
            fid = os.path.basename(self.inputs.outname)
            fname = fid + self.ext
        else:
            assert False
        return fname
