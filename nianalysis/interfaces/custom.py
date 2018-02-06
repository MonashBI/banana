from __future__ import absolute_import
from nipype.interfaces.base import (BaseInterface, BaseInterfaceInputSpec,
                                    traits, File, TraitedSpec, Directory)
import nibabel as nib
import numpy as np
from nipype.utils.filemanip import split_filename
import os
import matplotlib.pyplot as plot
from sklearn.decomposition import PCA
import glob
import shutil
import dicom


class PETdrInputSpec(BaseInterfaceInputSpec):

    volume = File(exists=True, desc='4D input for the dual regression',
                  mandatory=True)
    regression_map = File(exists=True, desc='3D map to use for the spatial '
                          'regression (first step of the dr)', mandatory=True)
    threshold = traits.Float(desc='Threshold to be applied to the abs(reg_map)'
                             ' before regression (default zero)', default=0.0)
    binarize = traits.Bool(desc='If True, all the voxels greater than '
                           'threshold will be set to 1 (default False)',
                           default=False)


class PETdrOutputSpec(TraitedSpec):

    spatial_map = File(
        exists=True, desc='Nifti file containing result for the temporal '
        'regression')
    timecourse = File(
        exists=True, desc='Png file containing result for the spatial '
        'regression')


class PETdr(BaseInterface):

    input_spec = PETdrInputSpec
    output_spec = PETdrOutputSpec

    def _run_interface(self, runtime):
        fname = self.inputs.volume
        mapname = self.inputs.regression_map
        th = self.inputs.threshold
        binarize = self.inputs.binarize
        _, base, _ = split_filename(fname)
        _, base_map, _ = split_filename(mapname)

        img = nib.load(fname)
        data = np.array(img.get_data())
        spatial_regressor = nib.load(mapname)
        spatial_regressor = np.array(spatial_regressor.get_data())

        n_voxels = data.shape[0]*data.shape[1]*data.shape[2]
        ts = data.reshape(n_voxels, data.shape[3])
        mask = spatial_regressor.reshape(n_voxels, 1)
        if th and not binarize:
            mask[np.abs(mask) < th] = 0
            base = base+'_th_{}'.format(str(th))
        elif th and binarize:
            mask[mask < th] = 0
            mask[mask >= th] = 1
            base = base+'_bin_th_{}'.format(str(th))
        timecourse = np.dot(ts.T, mask)
        sm = np.dot(ts, timecourse)
        mean = np.mean(sm)
        std = np.std(sm)
        sm_zscore = (sm-mean)/std

        im2save = nib.Nifti1Image(
            sm_zscore.reshape(spatial_regressor.shape), affine=img.affine)
        nib.save(
            im2save, '{0}_{1}_GLM_fit_zscore.nii.gz'.format(base, base_map))

        plot.plot(timecourse)
        plot.savefig('{0}_{1}_timecourse.png'.format(base, base_map))
        plot.close()

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        fname = self.inputs.volume
        th = self.inputs.threshold
        binarize = self.inputs.binarize
        mapname = self.inputs.regression_map

        _, base_map, _ = split_filename(mapname)
        _, base, _ = split_filename(fname)
        if th and not binarize:
            base = base+'_th_{}'.format(str(th))
        elif th and binarize:
            base = base+'_bin_th_{}'.format(str(th))

        outputs["spatial_map"] = os.path.abspath(
            '{0}_{1}_GLM_fit_zscore.nii.gz'.format(base, base_map))
        outputs["timecourse"] = os.path.abspath(
            '{0}_{1}_timecourse.png'.format(base, base_map))

        return outputs


class GlobalTrendRemovalInputSpec(BaseInterfaceInputSpec):

    volume = File(exists=True, desc='4D input file',
                  mandatory=True)


class GlobalTrendRemovalOutputSpec(TraitedSpec):

    detrended_file = File(
        exists=True, desc='4D file with the first temporal PCA component'
        ' removed')


class GlobalTrendRemoval(BaseInterface):

    input_spec = GlobalTrendRemovalInputSpec
    output_spec = GlobalTrendRemovalOutputSpec

    def _run_interface(self, runtime):

        fname = self.inputs.volume
        _, base, _ = split_filename(fname)

        img = nib.load(fname)
        data = np.array(img.get_data())

        n_voxels = data.shape[0]*data.shape[1]*data.shape[2]
        ts = data.reshape(n_voxels, data.shape[3])
        pca = PCA(50)
        pca.fit(ts)
        baseline = np.reshape(
            pca.components_[0, :], (len(pca.components_[0, :]), 1))
        new_ts = ts.T-np.dot(baseline, np.dot(np.linalg.pinv(baseline), ts.T))
        im2save = nib.Nifti1Image(
            new_ts.T.reshape(data.shape), affine=img.affine)
        nib.save(
            im2save, '{}_baseline_removed.nii.gz'.format(base))

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        fname = self.inputs.volume

        _, base, _ = split_filename(fname)

        outputs["detrended_file"] = os.path.abspath(
            '{}_baseline_removed.nii.gz'.format(base))

        return outputs


class SUVRCalculationInputSpec(BaseInterfaceInputSpec):

    volume = File(exists=True, desc='3D input file',
                  mandatory=True)
    base_mask = File(exists=True, desc='3D baseline mask',
                     mandatory=True)


class SUVRCalculationOutputSpec(TraitedSpec):

    SUVR_file = File(
        exists=True, desc='3D SUVR file')


class SUVRCalculation(BaseInterface):

    input_spec = SUVRCalculationInputSpec
    output_spec = SUVRCalculationOutputSpec

    def _run_interface(self, runtime):

        fname = self.inputs.volume
        maskname = self.inputs.base_mask
        _, base, _ = split_filename(fname)

        img = nib.load(fname)
        data = np.array(img.get_data())
        mask = nib.load(maskname)
        mask = np.array(mask.get_data())

        [x, y, z] = np.where(mask > 0)
        ii = np.arange(x.shape[0])
        mean_uptake = np.mean(data[x[ii], y[ii], z[ii]])
        new_data = data / mean_uptake
        im2save = nib.Nifti1Image(new_data, affine=img.affine)
        nib.save(im2save, '{}_SUVR.nii.gz'.format(base))

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        fname = self.inputs.volume

        _, base, _ = split_filename(fname)

        outputs["SUVR_file"] = os.path.abspath(
            '{}_SUVR.nii.gz'.format(base))

        return outputs


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

        concat = reg_mat[:]
        self.gen_motion_mat(concat, qform_mat, out_name)
        os.mkdir(out_name)
        mm = glob.glob('*motion_mat*.mat')
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


class DicomHeaderInfoExtractionInputSpec(BaseInterfaceInputSpec):

    dicom_folder = Directory(exists=True, desc='Directory with DICOM files',
                             mandatory=True)
    multivol = traits.Bool(desc='Specify whether a scan is 3D or 4D',
                           default=False)


class DicomHeaderInfoExtractionOutputSpec(TraitedSpec):

    tr = traits.Float(desc='Repetition time.')
    start_time = traits.Str(desc='Scan start time.')
    real_duration = traits.Str(desc='For 4D files, this will be the number of '
                               'volumes multiplied by the TR.')
    tot_duration = traits.Str(
        desc='Scan duration as extracted from the header.')
    ped = traits.Str(desc='Phase encoding direction.')
    phase_offset = traits.Str(desc='Phase offset.')


class DicomHeaderInfoExtraction(BaseInterface):

    input_spec = DicomHeaderInfoExtractionInputSpec
    output_spec = DicomHeaderInfoExtractionOutputSpec

    def _run_interface(self, runtime):

        list_dicom = sorted(glob.glob(self.inputs.dicom_folder+'/*'))
        multivol = self.inputs.multivol
        ped = ''
        phase_offset = ''
        self.dict_output = {}

        with open(list_dicom[0], 'r') as f:
            for line in f:
                if 'TotalScan' in line:
                    total_duration = line.split('=')[-1].strip()
                    if not multivol:
                        real_duration = total_duration
                elif 'alTR[0]' in line:
                    tr = float(line.split('=')[-1].strip())/1000000
                elif 'SliceArray.asSlice[0].dInPlaneRot' and multivol:
                    phase_offset = float(line.split('=')[-1].strip())
                    if np.abs(phase_offset) > 1 and np.abs(phase_offset) < 3:
                        ped = 'ROW'
                    elif np.abs(phase_offset) < 1 or np.abs(phase_offset) > 3:
                        ped = 'COL'
        if multivol:
            n_vols = len(list_dicom)
            real_duration = n_vols*tr

        hd = dicom.read_file(list_dicom[0])
        try:
            start_time = str(hd.AcquisitionTime)
        except AttributeError:
            try:
                start_time = str(hd.AcquisitionDateTime)[8:]
            except AttributeError:
                raise Exception('No acquisition time found for this scan.')
        self.dict_output['start_time'] = start_time
        self.dict_output['tr'] = tr
        self.dict_output['total_duration'] = total_duration
        self.dict_output['real_duration'] = real_duration
        self.dict_output['ped'] = ped
        self.dict_output['phase_offset'] = str(phase_offset)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()

        outputs["start_time"] = self.dict_output['start_time']
        outputs["tr"] = self.dict_output['tr']
        outputs["total_duration"] = self.dict_output['total_duration']
        outputs["real_duration"] = self.dict_output['real_duration']
        outputs["ped"] = self.dict_output['ped']
        outputs["phase_offset"] = self.dict_output['phase_offset']

        return outputs
