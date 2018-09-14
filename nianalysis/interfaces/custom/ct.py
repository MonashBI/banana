'''
    Created on 14 Sep. 2018

    '''
# This script is used to convert a CT image into a virtual U-map
import os.path
from nipype.interfaces.base import (
    TraitedSpec, BaseInterface, File, isdefined, traits)
import nibabel as nib
import numpy as np


class Ct2UmapInputSpec(TraitedSpec):
    ct_reg = File(mandatory=True, desc='registered ct 1 image file')
    a = traits.Float(desc="Enter values a: ")
    b = traits.Float(desc="Enter values b: ")
    BP= traits.Int(desc="the break point BP : ")
    sute_fix_template = File( genfile=True,desc='sute fixed map in template space')


class Ct2UmapOutputSpec(TraitedSpec):
    sute_fix_template = File(exists=True, desc='sute fixed map in template space')


class Ct2Umap(BaseInterface):
    '''
    The kVp‐dependent values a,b and the break point (BP) defining the
    transformation, from HU to 511 keV linear attenuation for soft tissue,
    and bone;
    The values for different energy: 80kev=[a=0.0000364,b=0.0626,BP=1050],
    100kev=[a=0.0000443,b=0.0544,BP=1052], 110kev=[a=0.0000492,b=0.0488,BP=1043], 120kev=[a=0.000051,b=0.0471,BP=1047], 130kev=[a=0.0000551,b=0.0424,BP=1037], 140kev=[a=0.0000564,b=0.0408,BP=1030];
        Example:input_image.nii.gz a b BP output_image_name
    '''

    input_spec = Ct2UmapInputSpec
    output_spec = Ct2UmapOutputSpec
        
    def _run_interface(self, runtime):
        ct = nib.load(self.inputs.ct_reg)
        # CONVERSION from HU to PET u values
        # Carney J P et al. (2006) Med. Phys. 33 976-83
        ct_data = ct.get_data()
        kt= np.array(ct_data)
        nans = np.isnan(kt)
        kt[nans] = 0
        
        low = kt < self.inputs.BP
        kt[low] = 0.000096 * (1000. + kt[low])
        high = kt >= self.inputs.BP
        kt[high] = self.inputs.a * (1000. + kt[high]) + self.inputs.b
        low_u =kt<0
        kt[low_u]=0
        
        # End of conversion
        
        umap=10000.*(kt)
        nans = np.isnan(umap)
        umap[nans] = 0
        save_im = nib.Nifti1Image(umap, affine=ct.affine)
        nib.save(save_im, self._gen_filename('sute_fix_template'))
        return runtime
    
    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['sute_fix_template'] = self._gen_sute_fix_template_fname()
        return outputs
                 
    def _gen_filename(self, name):
        if name == 'sute_fix_template':
            fname = self._gen_sute_fix_template_fname()
        else:
            assert False
        return fname
                 
    def _gen_sute_fix_template_fname(self):
        if isdefined(self.inputs.sute_fix_template):
            sute_fix_template_fname = self.inputs.sute_fix_template
        else:
            sute_fix_template_fname = os.path.join(os.getcwd(),"sute_fix_template.nii.gz")
        return sute_fix_template_fname
    
    
    
