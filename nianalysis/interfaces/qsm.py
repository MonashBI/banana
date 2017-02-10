from nipype.interfaces.matlab import MatlabCommand
from nipype.interfaces.base import TraitedSpec, BaseInterface, BaseInterfaceInputSpec, File, Directory
import os
from string import Template

class PrepareInputSpec(BaseInterfaceInputSpec):
    in_dir = Directory(exists=True, mandatory=True)
    out_dir = Directory('Raw/', usedefault=True)
    out_file = File('Raw/Raw_MAGNITUDE.nii.gz', usedefault=True)

class PrepareOutputSpec(TraitedSpec):
    out_dir = Directory(exists=True)
    out_file = File(exists=True)

class Prepare(BaseInterface):
    input_spec = PrepareInputSpec
    output_spec = PrepareOutputSpec

    def _run_interface(self, runtime):
        d = dict(in_dir=self.inputs.in_dir,
        out_dir=self.inputs.out_dir, 
        out_file=self.inputs.out_file)
        
        #this is your MATLAB code template
        script = Template("""in_dir = '$in_dir';
out_dir = '$out_dir';
out_file = '$out_file';
Prepare_Data(in_dir, out_dir, out_file);
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
        outputs['out_dir'] = os.path.abspath(self.inputs.out_dir)
        return outputs
    
    

class STIInputSpec(BaseInterfaceInputSpec):
    in_dir = File(exists=True, mandatory=True)
    mask_file = File(exists=True, mandatory=True)
    qsm = File('QSM.nii.gz', usedefault=True)
    tissue_phase = File('TissuePhase.nii.gz', usedefault=True)

class STIOutputSpec(TraitedSpec):
    qsm = File(exists=True)
    tissue_phase = File(exists=True)

class STI(BaseInterface):
    input_spec = STIInputSpec
    output_spec = STIOutputSpec

    def _run_interface(self, runtime):
        d = dict(in_dir=self.inputs.in_dir,
                 mask_file=self.inputs.mask_file,
                 tissue_phase=self.inputs.tissue_phase,
                 qsm=self.inputs.qsm)
        #this is your MATLAB code template
        script = Template("""in_dir = '$in_dir';
qsm_file = '$qsm';
mask_file = '$mask_file';
tissue_file = '$tissue_phase';
QSM_SingleEcho(in_dir, mask_file, qsm_file, tissue_file);
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
        outputs['qsm'] = os.path.abspath(self.inputs.qsm)
        outputs['tissue_phase'] = os.path.abspath(self.inputs.tissue_phase)
        return outputs