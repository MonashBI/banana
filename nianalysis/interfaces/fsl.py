import os.path
import warnings
from string import Template
from nibabel import load
from nipype.interfaces.base import (
    File, traits, TraitedSpec, BaseInterface, BaseInterfaceInputSpec)
from glob import glob
from nipype.interfaces.fsl.base import (FSLCommand, FSLCommandInputSpec, Info)
from nipype.interfaces.base import (load_template, File, traits, isdefined,
                                    TraitedSpec, BaseInterface, Directory,
                                    InputMultiPath, OutputMultiPath,
                                    BaseInterfaceInputSpec)

warn = warnings.warn
warnings.filterwarnings('always', category=UserWarning)

feat_template_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "resources", 'temp.fsf')


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
    feat_dir = Directory(exists=True,mandatory=True, argstr="%s",position=0,desc="Input feat preprocessed directory")
    train_data = File(exists=True,mandatory=True,argstr="%s",position=1,desc="Training file")
    component_threshold= traits.Int(argstr="%d",mandatory = True,position=2,desc="threshold for the number of components")
    motion_reg = traits.Bool(position=3, argstr='-m', desc="motion parameters regression")

class FSLFIXOutputSpec(TraitedSpec):
    output = File (exists=True,desc="cleaned output")
    

class FSLFIX(FSLCommand):
    
    _cmd = 'fix'
    input_spec = FSLFIXInputSpec
    output_spec = FSLFIXOutputSpec


    def _list_outputs(self):
        outputs = self.output_spec().get()
        print self.inputs.feat_dir+'./filtered_func_data_clean.nii*' 
        outputs['output'] = os.path.abspath(glob(self.inputs.feat_dir+'/filtered_func_data_clean.nii*')[0])
        return outputs
