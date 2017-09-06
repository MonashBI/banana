'''
Created on 1 Sep. 2017

@author: jakubb
'''

from __future__ import absolute_import
from nipype.interfaces import fsl
import os.path
from nipype.interfaces.base import (
    TraitedSpec, BaseInterface, File, Directory, traits)
import glob
import numpy as np
import dicom
import nibabel as nib
from multiprocessing import Pool

def umap_conv_unwarped(arg, **kwarg):

    return Nii2Dicom.umap_conv(*arg, **kwarg)


class Nii2DicomInputSpec(TraitedSpec):
    sute_cont_nii = File(mandatory=True, desc='sute continuous map nifti')
    sute_fix_nii = File(mandatory=True, desc='sute fixed map nifti')
    umap_ute_dir = Directory(mandatory=True, desc='original umaps')
    cpu_number = traits.Int(desc="cpu numbers", mandatory=True)


class Nii2DicomOutputSpec(TraitedSpec):
    sute_cont_dicom = Directory(exists=True, desc='sute continuous map in template space')
    sute_fix_dicom = Directory(exists=True, desc='sute fixed map in template space')



class Nii2Dicom(BaseInterface):
    """
    Creates two umaps in dicom format
    
    fully compatible with the UTE study:
    
    Attenuation Correction pipeline
    
    """

    input_spec = Nii2DicomInputSpec
    output_spec = Nii2DicomOutputSpec

    def _run_interface(self, runtime):
        list_umap = [self.inputs.sute_cont_nii, self.inputs.sute_fix_nii]
        umap_dcm = self.inputs.umap_ute_dir
        
        for j, umap in enumerate(list_umap):
            
            pt_name = os.path.join(os.getcwd(), umap.split('.')[0] + '_dicom')
            spl = fsl.Split()
            spl.inputs.dimension = 'z'
            spl.inputs.in_file = umap
            spl.run()
            
            list_nifti = sorted(glob.glob(os.getcwd()+'vol0*.ni*'))
            list_dcm = sorted(glob.glob(umap_dcm + '/*.dcm'))
            if os.path.isdir(pt_name) is False:
                os.mkdir(pt_name)
            
            ii = np.arange(len(list_dcm))
            ii = ii.tolist()
            p=Pool(self.inputs.cpu_number)
            p.map(umap_conv_unwarped, 
                  zip([self]*len(ii), [j]*len(ii), ii, list_dcm, list_nifti, [pt_name]*len(ii)))
            
            for f in list_nifti:
                os.remove(f)
        
        return runtime
    
    def umap_conv(self, j, i, f, list_nifti, pt_name):
        
        dcm = dicom.read_file(f)
        nifti = nib.load(list_nifti)
        nifti = nifti.get_data()
        nifti = nifti.astype('uint16')
        for n, val in enumerate(dcm.pixel_array.flat):

            dcm.pixel_array.flat[n] = nifti.flat[n]

        dcm.PixelData = dcm.pixel_array.T.tostring()
        
        dcm.save_as('{0}/{1}.dcm'
            .format(pt_name, str(i + 1).zfill(4)))
    
    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['sute_cont_dicom'] = os.path.join(os.getcwd(), 'sute_cont_dicom')
        outputs['sute_fix_dicom'] = os.path.join(os.getcwd(), 'sute_fix_dicom')
        return outputs
    
    
    