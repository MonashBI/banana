from nipype.interfaces.matlab import MatlabCommand
import nianalysis.interfaces
from nipype.interfaces.base import (
    TraitedSpec, BaseInterface, BaseInterfaceInputSpec, File, Directory)
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


class STIInputSpec(BaseInterfaceInputSpec):
    in_dir = Directory(exists=True, mandatory=True)
    mask_file = File(exists=True, mandatory=True)


class STIOutputSpec(TraitedSpec):
    qsm = File(exists=True)
    tissue_phase = File(exists=True)
    tissue_mask = File(exists=True)


class STI_DE(BaseInterface):
    input_spec = STIInputSpec
    output_spec = STIOutputSpec

    def _run_interface(self, runtime):  # @UnusedVariable
        self.working_dir = os.path.abspath(os.getcwd())
        script = (
            "addpath(genpath('{matlab_dir}'));\n"
            "QSM_DualEcho('{in_dir}', '{mask_file}', '{out_dir}');\n"
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
            'QSM',
            'TissuePhase.nii.gz')
        outputs['tissue_mask'] = os.path.join(
            self.working_dir,
            'QSM',
            'PhaseMask.nii.gz')
        return outputs

class STI(BaseInterface):
    input_spec = STIInputSpec
    output_spec = STIOutputSpec

    def _run_interface(self, runtime):  # @UnusedVariable
        self.working_dir = os.path.abspath(os.getcwd())
        script = (
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
