from nipype.interfaces.matlab import MatlabCommand
import nianalysis.interfaces
from nipype.interfaces.base import (
    TraitedSpec, traits, BaseInterface, BaseInterfaceInputSpec, File, Directory)
import os

class PrepareInputSpec(BaseInterfaceInputSpec):
    in_dir = Directory(exists=True, mandatory=True)


class PrepareOutputSpec(TraitedSpec):
    out_dir = Directory(exists=True)
    out_file = File(exists=True)


class Prepare(BaseInterface):
    input_spec = PrepareInputSpec
    output_spec = PrepareOutputSpec

    def _run_interface(self, runtime):  # @UnusedVariable
        self.working_dir = os.path.abspath(os.getcwd())
        script = (
            "set_param(0,'CharacterEncoding','UTF-8');\n"
            "addpath(genpath('{matlab_dir}'));\n"
            "Prepare_Raw_Channels('{in_dir}', '{out_dir}');\n"
            "exit;\n").format(
                in_dir=self.inputs.in_dir,
                out_dir=self.working_dir,
                matlab_dir=os.path.abspath(os.path.join(
                    os.path.dirname(nianalysis.interfaces.__file__),
                    'resources', 'matlab', 'qsm')))
        mlab = MatlabCommand(script=script, mfile=True)
        result = mlab.run()
        return result.runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_dir'] = os.path.join(self.working_dir, 'Raw')
        outputs['out_file'] = os.path.join(
            self.working_dir,
            'Raw',
            'Raw_MAGNITUDE.nii.gz')
        return outputs

class FillHolesInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True)

class FillHolesOutputSpec(TraitedSpec):
    out_file = File(desc='Filled mask file')


class FillHoles(BaseInterface):
    input_spec = FillHolesInputSpec
    output_spec = FillHolesOutputSpec

    def _run_interface(self, runtime):  # @UnusedVariable
        self.working_dir = os.path.abspath(os.getcwd())
        script = (
            "set_param(0,'CharacterEncoding','UTF-8');\n"
            "addpath(genpath('{matlab_dir}'));\n"
            "fillholes('{in_file}', '{out_file}');\n"
            "exit;\n").format(
                in_file=self.inputs.in_file,
                out_file=os.path.join(os.getcwd(),
                                         self._gen_filename('out_file')),
                matlab_dir=os.path.abspath(os.path.join(
                    os.path.dirname(nianalysis.interfaces.__file__),
                    'resources', 'matlab', 'qsm')))
        mlab = MatlabCommand(script=script, mfile=True)
        result = mlab.run()
        return result.runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = os.path.join(os.getcwd(),
                                         self._gen_filename('out_file'))
        return outputs
    
    def _gen_filename(self, name):
        if name == 'out_file':
            fname = 'Filled_Mask.nii.gz'
        else:
            assert False
        return fname

class CSVSummaryInputsSpec(BaseInterfaceInputSpec):
    in_ldn_mean = traits.List(traits.Float())
    in_ldn_std = traits.List(traits.Float())
    in_ldn_hist = traits.List(traits.List(traits.Float()))
    in_rdn_mean = traits.List(traits.Float())
    in_rdn_std = traits.List(traits.Float())
    in_rdn_hist = traits.List(traits.List(traits.Float()))
    
class CSVSummaryOutputSpec(TraitedSpec):
    out_file = File(exists=True)

class CSVSummary(BaseInterface):
    input_spec = CSVSummaryInputsSpec
    output_spec = CSVSummaryOutputSpec

    def _run_interface(self, runtime):  # @UnusedVariable
        with open(os.path.join(os.getcwd(),
                               self._gen_filename('out_file')), 'w') as fp:
            fp.write('ldn_mean, ldn_std, ldn_hist,'+
                'rdn_mean, rdn_std, rdn_hist'+
                '\n')
            for tple in zip(self.inputs.in_ldn_mean, self.inputs.in_ldn_std, self.inputs.in_ldn_hist, 
                            self.inputs.in_rdn_mean, self.inputs.in_rdn_std, self.inputs.in_rdn_hist):
                fp.write(','.join(str(t) for t in tple) + '\n')
        
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = os.path.join(os.getcwd(),
                                         self._gen_filename('out_file'))
        return outputs
    
    def _gen_filename(self, name):
        if name == 'out_file':
            fname = 'qsm_summary.csv'
        else:
            assert False
        return fname
    
class STIInputSpec(BaseInterfaceInputSpec):
    in_dir = Directory(exists=True, mandatory=True)
    mask_file = File(exists=True, mandatory=True)
    echo_times = traits.List(traits.Float(), value=[20.0], desc='Echo times in ms')

class STIOutputSpec(TraitedSpec):
    qsm = File(exists=True)
    tissue_phase = File(exists=True)
    tissue_mask = File(exists=True)

class STI(BaseInterface):
    input_spec = STIInputSpec
    output_spec = STIOutputSpec

    def _run_interface(self, runtime):  # @UnusedVariable
        self.working_dir = os.path.abspath(os.getcwd())
        script = (
            "set_param(0,'CharacterEncoding','UTF-8');\n"
            "addpath(genpath('{matlab_dir}'));\n"
            "QSM('{in_dir}', '{mask_file}', '{out_dir}', {echo_times});\n"
            "exit;").format(
                in_dir=self.inputs.in_dir,
                mask_file=self.inputs.mask_file,
                out_dir=self.working_dir,
                echo_times=self.inputs.echo_times,
                matlab_dir=os.path.abspath(os.path.join(
                    os.path.dirname(nianalysis.interfaces.__file__),
                    'resources', 'matlab', 'qsm')))
        mlab = MatlabCommand(script=script, mfile=True)
        result = mlab.run()
        return result.runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['qsm'] = os.path.join(self.working_dir, 'QSM', 'QSM.nii.gz')
        outputs['tissue_phase'] = os.path.join(
            self.working_dir,
            'QSM',
            'TissuePhase.nii.gz')
        outputs['tissue_mask'] = os.path.join(
            self.working_dir,
            'QSM',
            'PhaseMask.nii.gz')
        return outputs

class STI_SE(BaseInterface):
    input_spec = STIInputSpec
    output_spec = STIOutputSpec

    def _run_interface(self, runtime):  # @UnusedVariable
        self.working_dir = os.path.abspath(os.getcwd())
        script = (
            "set_param(0,'CharacterEncoding','UTF-8');\n"
            "addpath(genpath('{matlab_dir}'));\n"
            "QSM_SingleEcho('{in_dir}', '{mask_file}', '{out_dir}');\n"
            "exit;").format(
                in_dir=self.inputs.in_dir,
                mask_file=self.inputs.mask_file,
                out_dir=self.working_dir,
                matlab_dir=os.path.abspath(os.path.join(
                    os.path.dirname(nianalysis.interfaces.__file__),
                    'resources', 'matlab', 'qsm')))
        mlab = MatlabCommand(script=script, mfile=True)
        result = mlab.run()
        return result.runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['qsm'] = os.path.join(self.working_dir, 'QSM', 'QSM.nii.gz')
        outputs['tissue_phase'] = os.path.join(
            self.working_dir,
            'TissuePhase',
            'TissuePhase.nii.gz')
        outputs['tissue_mask'] = os.path.join(
            self.working_dir,
            'TissuePhase',
            'CoilMasks.nii.gz')
        return outputs
