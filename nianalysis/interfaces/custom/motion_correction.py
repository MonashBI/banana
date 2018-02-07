from __future__ import absolute_import
from nipype.interfaces.base import (BaseInterface, BaseInterfaceInputSpec,
                                    traits, File, TraitedSpec, Directory)
import numpy as np
from nipype.utils.filemanip import split_filename
import os
import glob
import shutil


class MotionMatCalculationInputSpec(BaseInterfaceInputSpec):

    reg_mat = File(exists=True, desc='Registration matrix', mandatory=True)
    qform_mat = File(exists=True, desc='Qform matrix', mandatory=True)
    align_mats = Directory(exists=True, desc='Directory with intra-scan '
                           'alignment matrices', default=None)


class MotionMatCalculationOutputSpec(TraitedSpec):

    motion_mats = Directory(exists=True, desc='Directory with resultin motion'
                            ' matrices')


class MotionMatCalculation(BaseInterface):

    input_spec = MotionMatCalculationInputSpec
    output_spec = MotionMatCalculationOutputSpec

    def _run_interface(self, runtime):

        reg_mat = np.loadtxt(self.inputs.reg_mat)
        qform_mat = np.loadtxt(self.inputs.qform_mat)
        _, out_name, _ = split_filename(self.inputs.reg_mat)
        if self.inputs.align_mats:
            list_mats = sorted(glob.glob(self.inputs.align_mats+'/MAT*'))
            if not list_mats:
                raise Exception(
                    'Folder {} is empty!'.format(self.inputs.align_mats))
            for mat in list_mats:
                m = np.loadtxt(mat)
                concat = np.dot(reg_mat, m)
                self.gen_motion_mat(concat, qform_mat, mat)
            mat_path, _, _ = split_filename(mat)
            mm = glob.glob(mat_path+'/*motion_mat*.mat')
        else:
            concat = reg_mat[:]
            self.gen_motion_mat(concat, qform_mat, out_name)
            mm = glob.glob('*motion_mat*.mat')
        os.mkdir(out_name)

        for f in mm:
            shutil.move(f, out_name)

        return runtime

    def gen_motion_mat(self, concat, qform, out_name):

        concat_inv = np.linalg.inv(concat)
        concat_inv_qform = np.dot(qform, concat_inv)
        concat_inv_qform_inv = np.linalg.inv(concat_inv_qform)
        np.savetxt('{0}_motion_mat_inv.mat'.format(out_name),
                   concat_inv_qform_inv)
        np.savetxt('{0}_motion_mat.mat'.format(out_name),
                   concat_inv_qform)

    def _list_outputs(self):
        outputs = self._outputs().get()

        _, out_name, _ = split_filename(self.inputs.reg_mat)

        outputs["motion_mats"] = os.path.abspath(out_name)

        return outputs


class MergeListMotionMatInputSpec(BaseInterfaceInputSpec):

    file_list = traits.List(mandatory=True, desc='List of files to merge')


class MergeListMotionMatOutputSpec(TraitedSpec):

    out_dir = Directory(desc='Output directory.')


class MergeListMotionMat(BaseInterface):

    input_spec = MergeListMotionMatInputSpec
    output_spec = MergeListMotionMatOutputSpec

    def _run_interface(self, runtime):

        file_list = self.inputs.file_list
        pth, _, _ = split_filename(file_list[0])
        if os.path.isdir(pth+'/motion_mats') is False:
            os.mkdir(pth+'/motion_mats')
        for f in file_list:
            shutil.copy(f, pth+'/motion_mats')

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()

        file_list = self.inputs.file_list
        pth, _, _ = split_filename(file_list[0])
        outputs["out_dir"] = pth+'/motion_mats'

        return outputs
