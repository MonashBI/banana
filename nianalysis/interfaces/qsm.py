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

class QSMSummaryInputsSpec(BaseInterfaceInputSpec):
    in_field_names = traits.List(traits.Str())
    in_field_values = traits.List(traits.List(traits.List(traits.Any())))
    in_visit_id = traits.List(traits.List(traits.Str()))
    in_subject_id = traits.List(traits.List(traits.Str()))
    
class QSMSummaryOutputSpec(TraitedSpec):
    out_file = File(exists=True)

class QSMSummary(BaseInterface):
    input_spec = QSMSummaryInputsSpec
    output_spec = QSMSummaryOutputSpec

    def _run_interface(self, runtime):  # @UnusedVariable
        with open(os.path.join(os.getcwd(),
                               self._gen_filename('out_file')), 'w') as fp:
            
            fp.write('subjectId,visitId,' + ','.join(str(t) for t in self.inputs.in_field_names) + '\n')
                
            for s, v, o in zip(self.inputs.in_subject_id, 
                               self.inputs.in_visit_id, 
                               self.inputs.in_field_values):
                for ts, tv, to in zip(s, v, o):
                    fp.write(','.join(str(t) for t in [ts, tv]) + ',')
                    fp.write(','.join(str(t) for t in to) + '\n')
        print os.path.join(os.getcwd(),
                               self._gen_filename('out_file'))
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
