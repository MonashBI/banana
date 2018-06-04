'''
Created on 1 Sep. 2017

@author: jakubb
'''


import os.path
from nipype.interfaces.base import (
    TraitedSpec, BaseInterface, File, isdefined)
import nibabel as nib
import numpy as np


class CoreUmapCalcInputSpec(TraitedSpec):
    air__mask = File(mandatory=True, desc='air mask file')
    bones__mask = File(mandatory=True, desc='bone mask file')
    ute1_reg = File(mandatory=True, desc='ute echo 1 image file')
    ute2_reg = File(mandatory=True, desc='ute echo 2 image file')
    sute_cont_template = File(
        genfile=True,
        desc='sute continuous map in template space')
    sute_fix_template = File(
        genfile=True,
        desc='sute fixed map in template space')


class CoreUmapCalcOutputSpec(TraitedSpec):
    sute_cont_template = File(
        exists=True,
        desc='sute continuous map in template space')
    sute_fix_template = File(
        exists=True,
        desc='sute fixed map in template space')


class CoreUmapCalc(BaseInterface):
    """Creates two umaps in the template space"""

    input_spec = CoreUmapCalcInputSpec
    output_spec = CoreUmapCalcOutputSpec

    def _run_interface(self, runtime):
        air = nib.load(self.inputs.air__mask)
        bones = nib.load(self.inputs.bones__mask)
        ute1 = nib.load(self.inputs.ute1_reg)
        ute2 = nib.load(self.inputs.ute2_reg)

        r2star_map = 1000. * (np.log(np.array(ute1.get_data())) 
                              - np.log(np.array(ute2.get_data()))) / 2.39
        nans = np.isnan(r2star_map)
        r2star_map[nans] = 0

        u_bone = 0.000001351 * (r2star_map**3) - 0.003617 * (r2star_map**2) + 3.841 * r2star_map - 19.46
        # CONVERSION from HU to PET u values
        # Conversion based on:
        # Carney J P et al. (2006) Med. Phys. 33 976-83
        
        a = 0.000051
        b = 0.0471
        BP = 1047.

        low = u_bone < BP
        u_bone[low] = 0.000096 * (1000. + u_bone[low])
        high = u_bone >= BP
        u_bone[high] = a * (1000. + u_bone[high]) + b

        low_u = u_bone < 0.1134
        u_bone[low_u] = 0.1134
        # End of conversion
        
        u_soft_fixed = 0.1
        u_air = 0.

        umap = 10000. * (u_air * np.array(air.get_data()) +
                         u_bone * np.array(bones.get_data()) +
                         (1 - np.array(bones.get_data())) *
                         (1 - np.array(air.get_data())) * u_soft_fixed)

        nans = np.isnan(umap)
        umap[nans] = 0
        save_im = nib.Nifti1Image(umap, affine=ute1.affine)
        nib.save(save_im, self._gen_filename('sute_cont_template'))

        u_bone2 = 0.151
        umap2 = 10000. * (u_air * np.array(air.get_data()) +
                         u_bone2 * np.array(bones.get_data()) +
                         (1 - np.array(bones.get_data())) *
                         (1 - np.array(air.get_data())) * u_soft_fixed)

        nans = np.isnan(umap2)
        umap2[nans] = 0
        save_im = nib.Nifti1Image(umap2, affine=ute1.affine)
        nib.save(save_im, self._gen_filename('sute_fix_template'))

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['sute_cont_template'] = self._gen_sute_cont_template_fname()
        outputs['sute_fix_template'] = self._gen_sute_fix_template_fname()
        return outputs

    def _gen_filename(self, name):
        if name == 'sute_cont_template':
            fname = self._gen_sute_cont_template_fname()
        elif name == 'sute_fix_template':
            fname = self._gen_sute_fix_template_fname()
        else:
            assert False
        return fname

    def _gen_sute_cont_template_fname(self):
        if isdefined(self.inputs.sute_cont_template):
            sute_cont_template_fname = self.inputs.sute_cont_template
        else:
            sute_cont_template_fname = os.path.join(
                os.getcwd(),
                "sute_cont_template.nii.gz")
        return sute_cont_template_fname

    def _gen_sute_fix_template_fname(self):
        if isdefined(self.inputs.sute_fix_template):
            sute_fix_template_fname = self.inputs.sute_fix_template
        else:
            sute_fix_template_fname = os.path.join(
                os.getcwd(),
                "sute_fix_template.nii.gz")
        return sute_fix_template_fname
