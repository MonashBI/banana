
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
from nipype.interfaces.base import isdefined
import nibabel as nib
import numpy as np
import ast
import subprocess as sp
import scipy
from random import shuffle
import shutil


warn = warnings.warn
warnings.filterwarnings('always', category=UserWarning)

feat_template_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "resources", 'temp.fsf')
optiBET_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                            'resources', 'bash', 'optiBET.sh'))

logger = logging.getLogger('arcana')


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
        d['outputdir'] = self.inputs.output_dir
        d['tr'] = self.inputs.tr
        d['volumes'] = self.get_vols(self.inputs.bold)
        d['del_volume'] = 0
        d['brain_thresh'] = self.inputs.brain_thresh
        d['epi_dwelltime'] = self.inputs.dwell_time
        d['epi_te'] = self.inputs.te
        d['unwarp_dir'] = self.inputs.unwarp_dir
        d['smoothing_fwhm'] = self.inputs.sfwhm
        d['highpass_cutoff'] = self.inputs.high_pass
        d['fmri_file'] = self.inputs.bold
        d['fmri_reference'] = self.inputs.bold_ref
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


class SignalRegressionInputSpec(BaseInterfaceInputSpec):

    fix_dir = Directory(exists=True, desc="Prepared FIX directory",
                        mandatory=True)
    labelled_components = File(
        exists=True, mandatory=True,
        desc=("Text file with FIX classified components."))
    motion_regression = traits.Bool(desc='Regress 24 motion paremeters.',
                                    default=False)
    highpass = traits.Float(
        desc='Whether or not to high pass the motion parameters', default=None)
    customRegressors = File(exists=True, default=None,
                            desc='File containing custom regressors.')


class SignalRegressionOutputSpec(TraitedSpec):

    output = File(exists=True, desc="cleaned output")


class SignalRegression(BaseInterface):

    input_spec = SignalRegressionInputSpec
    output_spec = SignalRegressionOutputSpec

    def _run_interface(self, runtime):

        im2filt = self.inputs.fix_dir+'/filtered_func_data.nii.gz'
        ref = nib.load(im2filt)
        components = []
        with open(self.inputs.labelled_components, 'r') as f:
            for line in f:
                components.append(line)
        bad_components = ast.literal_eval(components[-1].strip())
        bad_components = [x-1 for x in bad_components]
        im2filt = nib.load(im2filt)
        if self.inputs.highpass:
            hdr = im2filt.header
            sa = hdr.structarr
            TR = sa['pixdim'][4]
            print('Repetition time from the header: {} sec'.format(str(TR)))
        else:
            TR = None
        im2filt = im2filt.get_data()
        [x, y, z, t] = im2filt.shape
        im2filt = np.reshape(im2filt, (x*y*z, t), order='F').T
        ICA = self.normalise(np.loadtxt(self.inputs.fix_dir+'/melodic_mix'))
        if self.inputs.motion_regression:
            mp = self.inputs.fix_dir+'/mc/prefiltered_func_data_mcf.par'
            motion_confounds = self.create_motion_confounds(
                mp, self.inputs.highpass, TR=TR)
            ICA = ICA - (np.dot(motion_confounds,
                                np.dot(np.linalg.pinv(motion_confounds), ICA)))
            im2filt = (im2filt-np.dot(
                motion_confounds, np.dot(np.linalg.pinv(motion_confounds),
                                         im2filt)))
        if self.inputs.customRegressors:
            cr = np.loadtxt(self.inputs.customRegressors)
            if cr.shape[0] != t:
                print (
                    'custom regressors and input image have a different '
                    'time lenght. They will not be used for the regression.')
            else:
                cr = self.normalise(cr)
                im2filt = im2filt - np.dot(cr, np.dot(np.linalg.pinv(cr),
                                                      im2filt))

        betaICA = np.dot(np.linalg.pinv(ICA), im2filt)
        im2filt = im2filt - np.dot(ICA[:, bad_components],
                                   betaICA[bad_components, :])
        im_filt = np.reshape(im2filt.T, (x, y, z, t), order='F')
        im2save = nib.Nifti1Image(im_filt, affine=ref.get_affine())
        nib.save(
            im2save, self.inputs.fix_dir+'/filtered_func_data_clean.nii.gz')

        return runtime

    def create_motion_confounds(self, mp, hp, TR=None):

        confounds = np.loadtxt(mp)
        confounds = self.normalise(
            np.hstack((confounds, np.vstack((np.zeros(6),
                                             np.diff(confounds, axis=0))))))
        confounds = self.normalise(
            np.hstack((confounds, np.square(confounds))))
        if hp == 0:
            confounds = scipy.signal.detrend(confounds, axis=0, type='linear')
        elif hp > 0:
            im2save = nib.Nifti1Image(
                np.reshape(confounds.T, (confounds.shape[1], 1, 1,
                                         confounds.shape[0]), order='F'),
                affine=np.eye(4))
            nib.save(im2save, self.inputs.fix_dir+'/mc/mc_par_conf.nii.gz')
            cmd = (
                'fslmaths {0}/mc_par_conf.nii.gz -bptf {1} -1 '
                '{0}/mc_par_conf_hp'.format(self.inputs.fix_dir+'/mc',
                                            str(0.5*float(hp)/TR)))
            sp.check_output(cmd, shell=True)
            confounds_hp = (nib.load(
                self.inputs.fix_dir+'/mc/mc_par_conf_hp.nii.gz')).get_data()
            confounds = self.normalise(
                np.reshape(confounds_hp, (confounds.shape[1],
                                          confounds.shape[0]), order='F').T)

        return confounds

    def normalise(self, params):

        params = (params - np.mean(params, axis=0))/np.std(params, axis=0,
                                                           ddof=1)
        return params

    def _list_outputs(self):

        outputs = self._outputs().get()
        outputs['output'] = (
            self.inputs.fix_dir+'/filtered_func_data_clean.nii.gz')
        return outputs


class FSLFIXInputSpec(FSLCommandInputSpec):
    _xor_parameters = ('classification', 'regression', 'all')
    _xor_input_files = ('feat_dir', 'labelled_component')
    feat_dir = Directory(
        exists=True, argstr="%s", position=1, xor=_xor_input_files,
        desc="Input melodic preprocessed directory")
    train_data = File(exists=True, argstr="%s", position=2,
                      desc="Training file")
    regression = traits.Bool(desc='Regress previously classified components.',
                             position=0, argstr="-a", xor=_xor_parameters)
    classification = traits.Bool(
        desc='Components classification without regression.', position=0,
        argstr="-c", xor=_xor_parameters)
    all = traits.Bool(
        desc='Components classification and regression.', position=0,
        argstr="-f", xor=_xor_parameters)
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
        # print(self.inputs.feat_dir+'./filtered_func_data_clean.nii*')
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


class PrepareFIXTrainingInputSpec(BaseInterfaceInputSpec):

    inputs_list = traits.List()
    epi_number = traits.Int()


class PrepareFIXTrainingOutputSpec(TraitedSpec):

    prepared_dirs = traits.List()


class PrepareFIXTraining(BaseInterface):

    input_spec = PrepareFIXTrainingInputSpec
    output_spec = PrepareFIXTrainingOutputSpec

    def _run_interface(self, runtime):
        
        epi_number = self.inputs.epi_number
        inputs = self.inputs.inputs_list
        fix_dirs = [inputs[x:x+epi_number] for x in
                    np.arange(0, len(inputs)/2, epi_number, dtype=int)]
        labels = [inputs[x:x+epi_number] for x in
                    np.arange(len(inputs)/2, len(inputs), epi_number, dtype=int)]
        self.out_dirs = []
        

        if len(labels) != len(fix_dirs):
            raise Exception('The number of subjects provided is different from'
                            'the number of hand_label_noise files. Fix '
                            'training cannot be performed. Please check.')
        for i, label in enumerate(labels):
            for j in range(epi_number):
                with open(label[j], 'r') as f:
                    if 'not_provided' not in f.readline():
                        shutil.copy2(label[j], fix_dirs[i][j]+'/hand_labels_noise.txt')
                        self.out_dirs.append(fix_dirs[i][j])
        if not self.out_dirs:
            raise Exception('No non-empty hand_labels_noise.txt file found in the fix_dir '
                            'provided. In order to run FIX training at least 25 '
                            'hand_labels_noise.txt files have to be provided. Please '
                            'go through 25 single-subject MELODIC ICA results, create '
                            'those text files and upload them on XNAT. Have a look at '
                            'the documentation to have more information.')

        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        # print self.inputs.feat_dir+'./filtered_func_data_clean.nii*'
        outputs['prepared_dirs'] = self.out_dirs
        return outputs
