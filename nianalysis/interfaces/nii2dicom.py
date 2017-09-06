'''
Created on 1 Sep. 2017

@author: jakubb
'''

from __future__ import absolute_import
import shutil
import os.path
from nipype.interfaces.base import (
    TraitedSpec, BaseInterface, File, Directory, traits, isdefined)
import dicom
import nibabel as nib
from nianalysis.utils import split_extension


def umap_conv_unwarped(arg, **kwarg):

    return Nii2Dicom.umap_conv(*arg, **kwarg)


class Nii2DicomInputSpec(TraitedSpec):
    in_file = File(mandatory=True, desc='input nifti file')
    reference_dicom = File(mandatory=True, desc='original umap')
    out_file = File(genfile=True, desc='the output dicom file')


class Nii2DicomOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='the output dicom file')


class Nii2Dicom(BaseInterface):
    """
    Creates two umaps in dicom format

    fully compatible with the UTE study:

    Attenuation Correction pipeline

    """

    input_spec = Nii2DicomInputSpec
    output_spec = Nii2DicomOutputSpec

    def _run_interface(self, runtime):
        dcm = dicom.read_file(self.inputs.reference_dicom)
        nifti = nib.load(self.inputs.in_file)
        nifti = nifti.get_data()
        nifti = nifti.astype('uint16')
        for n in range(len(dcm.pixel_array.flat)):
            dcm.pixel_array.flat[n] = nifti.flat[n]
        dcm.PixelData = dcm.pixel_array.T.tostring()
        dcm.save_as(self.inputs.out_file)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = self._gen_outfilename()
        return outputs

    def _gen_filename(self, name):
        if name == 'out_file':
            fname = self._gen_outfilename()
        else:
            assert False
        return fname

    def _gen_outfilename(self):
        if isdefined(self.inputs.out_file):
            fpath = self.inputs.out_file
        else:
            fname = (
                split_extension(os.path.basename(self.inputs.in_file))[0] +
                '_dicom')
            fpath = os.path.join(os.getcwd(), fname)
        return fpath


class CopyToDicomDirInputSpec(TraitedSpec):
    in_files = File(mandatory=True, desc='input dicom files')
    out_dir = File(genfile=True, desc='the output dicom file')


class CopyToDicomDirOutputSpec(TraitedSpec):
    out_dir = Directory(exists=True, desc='the output dicom directory')


class CopyToDicomDir(BaseInterface):
    """
    Creates two umaps in dicom format

    fully compatible with the UTE study:

    Attenuation Correction pipeline

    """

    input_spec = CopyToDicomDirInputSpec
    output_spec = CopyToDicomDirOutputSpec

    def _run_interface(self, runtime):
        dirname = self._gen_outdirname()
        os.makedirs(dirname)
        for i, f in enumerate(self.inputs.in_files):
            fname = os.path.join(dirname, str(i).zfill(4)) + '.dcm'
            shutil.copy(f, fname)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_dir'] = self._gen_outfilename()
        return outputs

    def _gen_filename(self, name):
        if name == 'out_dir':
            fname = self._gen_outdirname()
        else:
            assert False
        return fname

    def _gen_outdirname(self):
        if isdefined(self.inputs.out_file):
            dpath = self.inputs.out_file
        else:
            dpath = os.path.join(os.getcwd(), 'dicom_dir')
        return dpath


class ListDirInputSpec(TraitedSpec):
    directory = File(mandatory=True, desc='directory to read')


class ListDirOutputSpec(TraitedSpec):
    files = traits.List(File(exists=True),
                        desc='The files present in the directory')


class ListDir(BaseInterface):
    """
    Creates two umaps in dicom format

    fully compatible with the UTE study:

    Attenuation Correction pipeline

    """

    input_spec = ListDirInputSpec
    output_spec = ListDirOutputSpec

    def _run_interface(self, runtime):
        return runtime

    def _list_outputs(self):
        dname = self.inputs.directory
        outputs = self._outputs().get()
        outputs['files'] = sorted(
            os.path.join(dname, f)
            for f in os.listdir(dname)
            if os.path.isfile(os.path.join(dname, f)))
        return outputs
