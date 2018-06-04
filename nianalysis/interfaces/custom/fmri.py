
from nipype.interfaces.base import (
    BaseInterface, BaseInterfaceInputSpec, TraitedSpec, Directory, File)
import os
import shutil


class PrepareFIXInputSpec(BaseInterfaceInputSpec):

    melodic_dir = Directory()
    filtered_epi = File(exists=True)
    t1_brain = File(exists=True)
    mc_par = File(exists=True)
    epi_brain_mask = File(exists=True)
    epi_preproc = File(exists=True)
    epi2t1_mat = File(exists=True)
    t12epi_mat = File(exists=True)
    t12MNI_mat = File(exists=True)
    MNI2t1_mat = File(exists=True)
    epi_mean = File(exists=True)


class PrepareFIXOutputSpec(TraitedSpec):

    fix_dir = Directory()


class PrepareFIX(BaseInterface):

    input_spec = PrepareFIXInputSpec
    output_spec = PrepareFIXOutputSpec

    def _run_interface(self, runtime):

        melodic_dir = self.inputs.melodic_dir
        filtered_epi = self.inputs.filtered_epi
        t1_brain = self.inputs.t1_brain
        mc_par = self.inputs.mc_par
        epi_brain_mask = self.inputs.epi_brain_mask
        epi_preproc = self.inputs.epi_preproc
        epi2t1_mat = self.inputs.epi2t1_mat
        t12epi_mat = self.inputs.t12epi_mat
        t12MNI_mat = self.inputs.t12MNI_mat
        MNI2t1_mat = self.inputs.MNI2t1_mat
        epi_mean = self.inputs.epi_mean

        shutil.copytree(melodic_dir, 'melodic_ica')
        os.mkdir('melodic_ica/reg')
        shutil.copy2(t12MNI_mat, 'melodic_ica/reg/highres2std.mat')
        shutil.copy2(MNI2t1_mat, 'melodic_ica/reg/std2highres.mat')
        shutil.copy2(epi2t1_mat, 'melodic_ica/reg/example_func2highres.mat')
        shutil.copy2(t1_brain, 'melodic_ica/reg/highres.nii.gz')
        shutil.copy2(epi_preproc, 'melodic_ica/reg/example_func.nii.gz')
        shutil.copy2(t12epi_mat, 'melodic_ica/reg/highres2example_func.mat')
        os.mkdir('melodic_ica/mc')
        shutil.copy2(mc_par, 'melodic_ica/mc/prefiltered_func_data_mcf.par')
        shutil.copy2(epi_brain_mask, 'melodic_ica/mask.nii.gz')
        shutil.copy2(epi_mean, 'melodic_ica/mean_func.nii.gz')
        shutil.copytree(melodic_dir, 'melodic_ica/filtered_func_data.ica')
        shutil.copy2(filtered_epi, 'melodic_ica/filtered_func_data.nii.gz')

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()

        outputs["fix_dir"] = os.getcwd()+'/melodic_ica'

        return outputs
