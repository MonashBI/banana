from nipype.interfaces.matlab import MatlabCommand
import banana.interfaces
from nipype.interfaces.base import (
    TraitedSpec, traits, BaseInterface, BaseInterfaceInputSpec, File,
    Directory)
import os
import os.path as op


SCRIPT_TEMPLATE = (
    "set_param(0,'CharacterEncoding','UTF-8');\n"
    "addpath(genpath('{}'));\n"
    "{{cmd}};\n"
    "exit;\n").format(op.join(banana.interfaces.RESOURCES_DIR, 'matlab',
                              'qsm'))


def matlab_cmd(cmd):
    return MatlabCommand(script=SCRIPT_TEMPLATE.format(cmd=cmd), mfile=True)


class ShMRFInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True)
    mask_file = File(exists=True, mandatory=True)


class ShMRFOutputSpec(TraitedSpec):
    out_file = File(exists=True)


class ShMRF(BaseInterface):
    input_spec = ShMRFInputSpec
    output_spec = ShMRFOutputSpec

    def _run_interface(self, runtime):  # @UnusedVariable
        mlab = matlab_cmd(
            "ShMRF('{in_file}', '{mask_file}', '{out_file}')").format(
                in_file=self.inputs.in_file,
                mask_file=self.inputs.mask_file,
                out_file=self._gen_filename('out_file'))
        result = mlab.run()
        return result.runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = self._gen_filename('out_file')

        return outputs

    def _gen_filename(self, name):
        if name == 'out_file':
            fname = 'ShMRF_Vein_Mask.nii.gz'
        else:
            assert False
        return op.abspath(fname)


class FlipSWIInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True)
    hdr_file = File(exists=True, mandatory=True)


class FlipSWIOutputSpec(TraitedSpec):
    out_file = File(exists=True)


class FlipSWI(BaseInterface):
    input_spec = FlipSWIInputSpec
    output_spec = FlipSWIOutputSpec

    def _run_interface(self, runtime):  # @UnusedVariable
        mlab = matlab_cmd(
            "FlipSWI('{in_file}', '{hdr_file}', '{out_file}')".format(
                in_file=self.inputs.in_file,
                hdr_file=self.inputs.hdr_file,
                out_file=self._gen_filename('out_file')))
        result = mlab.run()
        return result.runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = self._gen_filename('out_file')

        return outputs

    def _gen_filename(self, name):
        if name == 'out_file':
            fname = 'Flipped_Scanner_Image.nii.gz'
        else:
            assert False
        return op.abspath(fname)


class CVImageInputSpec(BaseInterfaceInputSpec):
    qsm = File(exists=True, mandatory=True)
    swi = File(exists=True, mandatory=True)
    mask = File(exists=True, mandatory=True)
    vein_atlas = File(exists=True, mandatory=True)
    q_prior = File(exists=True, mandatory=True)
    s_prior = File(exists=True, mandatory=True)
    a_prior = File(exists=True, mandatory=True)


class CVImageOutputSpec(TraitedSpec):
    out_file = File(exists=True)


class CVImage(BaseInterface):
    input_spec = CVImageInputSpec
    output_spec = CVImageOutputSpec

    def _run_interface(self, runtime):  # @UnusedVariable
        mlab = matlab_cmd(
            "CVImage('{qsm_file}', '{swi_file}', '{vein_atlas_file}', "
            "'{mask_file}', '{q_prior_file}', '{s_prior_file}', "
            "'{a_prior_file}', '{out_file}')".format(
                qsm_file=self.inputs.qsm,
                swi_file=self.inputs.swi,
                vein_atlas_file=self.inputs.vein_atlas,
                mask_file=self.inputs.mask,
                q_prior_file=self.inputs.q_prior,
                s_prior_file=self.inputs.s_prior,
                a_prior_file=self.inputs.a_prior,
                out_file=self._gen_filename('out_file')))
        result = mlab.run()
        return result.runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = self._gen_filename('out_file')
        return outputs

    def _gen_filename(self, name):
        if name == 'out_file':
            fname = 'CVImage.nii.gz'
        else:
            assert False
        return op.abspath(fname)


class PrepareInputSpec(BaseInterfaceInputSpec):
    in_dir = Directory(exists=True, mandatory=True)
    base_filename = traits.Str(
        value='T2swi3d_ axial_p2_0.9_iso_COSMOS_Straight_Coil',
        mandatory=True, desc='Base filename of coil files')
    echo_times = traits.List(traits.Float(), mandatory=True, value=[20.0],
                             desc='Echo times in ms')
    num_channels = traits.Int(value=32, mandatory=True,
                              desc='Number of channels')


class PrepareOutputSpec(TraitedSpec):
    out_dir = Directory(exists=True)
    out_file_fe = File(exists=True)
    out_file_le = File(exists=True)


class Prepare(BaseInterface):
    input_spec = PrepareInputSpec
    output_spec = PrepareOutputSpec

    def _run_interface(self, runtime):  # @UnusedVariable
        mlab = matlab_cmd(
            "Prepare_Raw_Channels('{in_dir}', '{filename}', {echo_times}, "
            "{num_channels}, '{out_dir}', '{out_file_fe}', '{out_file_le}')"
            .format(
                in_dir=self.inputs.in_dir,
                filename=self.inputs.base_filename,
                out_dir=self._gen_filename('out_dir'),
                out_file_fe=self._gen_filename('out_file_fe'),
                out_file_le=self._gen_filename('out_file_le'),
                echo_times=self.inputs.echo_times,
                num_channels=self.inputs.num_channels))
        result = mlab.run()
        return result.runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_dir'] = self._gen_filename('out_dir')
        outputs['out_file_fe'] = self._gen_filename('out_file_fe')
        outputs['out_file_le'] = self._gen_filename('out_file_le')

        return outputs

    def _gen_filename(self, name):
        if name == 'out_file_fe':
            fname = op.join('Raw', 'Raw_MAGNITUDE_FirstEcho.nii.gz')
        elif name == 'out_file_le':
            fname = op.join('Raw', 'Raw_MAGNITUDE_LastEcho.nii.gz')
        elif name == 'out_dir':
            fname = 'Raw'
        else:
            assert False
        return op.abspath(fname)


class FillHolesInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True)


class FillHolesOutputSpec(TraitedSpec):
    out_file = File(desc='Filled mask file')


class FillHoles(BaseInterface):
    input_spec = FillHolesInputSpec
    output_spec = FillHolesOutputSpec

    def _run_interface(self, runtime):  # @UnusedVariable
        mlab = matlab_cmd(
            "fillholes('{in_file}', '{out_file}');\n"
            "exit;\n").format(
                in_file=self.inputs.in_file,
                out_file=self._gen_filename('out_file'))
        result = mlab.run()
        return result.runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = self._gen_filename('out_file')
        return outputs

    def _gen_filename(self, name):
        if name == 'out_file':
            fname = 'Filled_Mask.nii.gz'
        else:
            assert False
        return op.abspath(fname)


class FitMaskInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True)
    initial_mask_file = File(exists=True, mandatory=True)


class FitMaskOutputSpec(TraitedSpec):
    out_file = File(desc='Fitted mask file')


class FitMask(BaseInterface):
    input_spec = FitMaskInputSpec
    output_spec = FitMaskOutputSpec

    def _run_interface(self, runtime):  # @UnusedVariable
        mlab = matlab_cmd(
            "FitMask('{in_file}', '{initial_mask_file}', '{out_file}');\n"
            "exit;\n").format(
                in_file=self.inputs.in_file,
                initial_mask_file=self.inputs.initial_mask_file,
                out_file=op.join(os.getcwd(),
                                         self._gen_filename('out_file')))
        result = mlab.run()
        return result.runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = self._gen_filename('out_file')
        return outputs

    def _gen_filename(self, name):
        if name == 'out_file':
            fname = 'Fitted_Mask.nii.gz'
        else:
            assert False
        return op.abspath(fname)


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
        with open(op.join(os.getcwd(),
                               self._gen_filename('out_file')), 'w') as fp:

            fp.write('subjectId,visitId,' + ','.join(
                str(t) for t in self.inputs.in_field_names) + '\n')

            for s, v, o in zip(self.inputs.in_subject_id,
                               self.inputs.in_visit_id,
                               self.inputs.in_field_values):
                for ts, tv, to in zip(s, v, o):
                    fp.write(','.join(str(t) for t in [ts, tv]) + ',')
                    fp.write(','.join(str(t) for t in to) + '\n')
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = self._gen_filename('out_file')
        return outputs

    def _gen_filename(self, name):
        if name == 'out_file':
            fname = 'qsm_summary.csv'
        else:
            assert False
        return op.abspath(fname)
