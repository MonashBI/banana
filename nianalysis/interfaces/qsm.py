from nipype.interfaces.matlab import MatlabCommand
import nianalysis.interfaces
from nipype.interfaces.base import TraitedSpec, BaseInterface, BaseInterfaceInputSpec, File, Directory
import os
from string import Template

class PrepareInputSpec(BaseInterfaceInputSpec):
    in_dir = Directory(exists=True, mandatory=True)

class PrepareOutputSpec(TraitedSpec):
    out_dir = Directory(exists=True)
    out_file = File(exists=True)

class Prepare(BaseInterface):
    input_spec = PrepareInputSpec
    output_spec = PrepareOutputSpec

    def _run_interface(self, runtime):
        d = dict(in_dir=self.inputs.in_dir,
        matlab_dir=os.path.abspath(os.path.join(os.path.dirname(nianalysis.interfaces.__file__),'matlab', 'qsm')))
        
        #this is your MATLAB code template
        script = Template("""in_dir = '$in_dir';
addpath(genpath('$matlab_dir'));
Prepare_Raw_Channels(in_dir);
exit;
""").substitute(d)

# 

        # mfile = True  will create an .m file with your script and executed.
        # Alternatively
        # mfile can be set to False which will cause the matlab code to be
        # passed
        # as a commandline argument to the matlab executable
        # (without creating any files).
        # This, however, is less reliable and harder to debug
        # (code will be reduced to
        # a single line and stripped of any comments).

        mlab = MatlabCommand(script=script, mfile=True)
        result = mlab.run()
        return result.runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_dir'] = os.path.abspath('Raw/')
        outputs['out_file'] = os.path.abspath('Raw/Raw_MAGNITUDE.nii.gz')
        return outputs
    
    

class STIInputSpec(BaseInterfaceInputSpec):
    in_dir = Directory(exists=True, mandatory=True)
    mask_file = File(exists=True, mandatory=True)

class STIOutputSpec(TraitedSpec):
    qsm = File(exists=True)
    tissue_phase = File(exists=True)
    tissue_mask = File(exists=True)

class STI(BaseInterface):
    input_spec = STIInputSpec
    output_spec = STIOutputSpec

    def _run_interface(self, runtime):
        d = dict(in_dir=self.inputs.in_dir,
                 mask_file=self.inputs.mask_file,
                 matlab_dir=os.path.abspath(os.path.join(os.path.dirname(nianalysis.interfaces.__file__),'matlab', 'qsm')))
        #this is your MATLAB code template
        script = Template("""in_dir = '$in_dir';
mask_file = '$mask_file';
addpath(genpath('$matlab_dir'));
QSM_SingleEcho(in_dir, mask_file);
exit;
""").substitute(d)

        # mfile = True  will create an .m file with your script and executed.
        # Alternatively
        # mfile can be set to False which will cause the matlab code to be
        # passed
        # as a commandline argument to the matlab executable
        # (without creating any files).
        # This, however, is less reliable and harder to debug
        # (code will be reduced to
        # a single line and stripped of any comments).

        mlab = MatlabCommand(script=script, mfile=True)
        result = mlab.run()
        return result.runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['qsm'] = os.path.abspath('QSM/QSM.nii.gz')
        outputs['tissue_phase'] = os.path.abspath('TissuePhase/TissuePhase.nii.gz')
        outputs['tissue_mask'] = os.path.abspath('TissuePhase/CoilMasks.nii.gz')
        return outputs