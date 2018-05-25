from __future__ import absolute_import
from nipype.interfaces.base import (BaseInterface, BaseInterfaceInputSpec,
                                    traits, File, TraitedSpec, Directory)
import numpy as np
from nipype.utils.filemanip import split_filename
import os
import glob
import shutil
import nibabel as nib
from nipype.interfaces.base import isdefined
import scipy.ndimage.measurements as snm
import datetime as dt
import matplotlib
import matplotlib.pyplot as plot
from nipype.interfaces import fsl
import pydicom
import math


class MotionMatCalculationInputSpec(BaseInterfaceInputSpec):

    reg_mat = File(exists=True, desc='Registration matrix')
    qform_mat = File(exists=True, desc='Qform matrix')
    dummy_input = Directory(desc='Dummy input in order to make the reference '
                            'motion mat pipeline work')
    align_mats = Directory(exists=True, desc='Directory with intra-scan '
                           'alignment matrices', default=None)
    reference = traits.Bool(desc='If True, the pipeline will save just an '
                            'identity matrix (motion mats for reference scan)',
                            default=False)


class MotionMatCalculationOutputSpec(TraitedSpec):

    motion_mats = Directory(exists=True, desc='Directory with resultin motion'
                            ' matrices')


class MotionMatCalculation(BaseInterface):

    input_spec = MotionMatCalculationInputSpec
    output_spec = MotionMatCalculationOutputSpec

    def _run_interface(self, runtime):

        reference = self.inputs.reference
        dummy = self.inputs.dummy_input
        if reference:
            np.savetxt('reference_motion_mat.mat', np.eye(4))
            np.savetxt('reference_motion_mat_inv.mat', np.eye(4))
            out_name = 'ref_motion_mats'
            mm = glob.glob('*motion_mat*.mat')
        else:
            reg_mat = np.loadtxt(self.inputs.reg_mat)
            qform_mat = np.loadtxt(self.inputs.qform_mat)
            _, out_name, _ = split_filename(self.inputs.reg_mat)
            if self.inputs.align_mats:
                list_mats = sorted(glob.glob(self.inputs.align_mats+'/MAT*'))
                if not list_mats:
                    list_mats = sorted(glob.glob(
                        self.inputs.align_mats+'/*.mat'))
                    if not list_mats:
                        raise Exception(
                            'Folder {} is empty!'.format(
                                self.inputs.align_mats))
                for mat in list_mats:
                    m = np.loadtxt(mat)
                    concat = np.dot(reg_mat, m)
                    self.gen_motion_mat(concat, qform_mat, mat.split('.')[0])
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

        if self.inputs.reference:
            out_name = 'ref_motion_mats'
        else:
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


class PrepareDWIInputSpec(BaseInterfaceInputSpec):

    pe_dir = traits.Str(mandatory=True, desc='Phase encoding direction')
    phase_offset = traits.Str(mandatory=True, desc='phase offset')
    dwi = File(mandatory=True, exists=True)
    dwi1 = File(mandatory=True, exists=True)
    topup = traits.Bool(desc='Specify whether the PrepareDWI output will be'
                        'used for TOPUP distortion correction')


class PrepareDWIOutputSpec(TraitedSpec):

    pe = traits.Str(desc='Phase encoding direction.')
    main = File(desc='4D dwi scan for eddy.')
    secondary = File(desc='3D dwi scan for distortion correction.')
    pe_1 = traits.Str(
        desc='Phase encoding direction second dwi.')


class PrepareDWI(BaseInterface):

    input_spec = PrepareDWIInputSpec
    output_spec = PrepareDWIOutputSpec

    def _run_interface(self, runtime):

        self.dict_output = {}
        pe_dir = self.inputs.pe_dir
        phase_offset = float(self.inputs.phase_offset)
        topup = self.inputs.topup
        dwi = nib.load(self.inputs.dwi)
        dwi = dwi.get_data()
        dwi1 = nib.load(self.inputs.dwi1)
        dwi1 = dwi1.get_data()
        if np.sign(phase_offset) == 1:
            if pe_dir == 'ROW':
                self.dict_output['pe'] = 'RL'
            elif pe_dir == 'COL':
                self.dict_output['pe'] = 'AP'
        elif np.sign(phase_offset) == -1:
            if pe_dir == 'ROW':
                self.dict_output['pe'] = 'LR'
            elif pe_dir == 'COL':
                self.dict_output['pe'] = 'PA'
#         if pe_dir == 'ROW':
#             if np.sign(phase_offset) == 1:
#                 self.dict_output['pe'] = 'RL'
#             else:
#                 self.dict_output['pe'] = 'LR'
#         elif pe_dir == 'COL':
#             if phase_offset < 1:
#                 self.dict_output['pe'] = 'AP'
#             else:
#                 self.dict_output['pe'] = 'PA'
#         else:
#             raise Exception('Phase encoding direction cannot be establish by'
#                             ' looking at the header. DWI pre-processing will'
#                             ' not be performed.')
        self.dict_output['pe_1'] = self.dict_output['pe'][::-1]

        if len(dwi.shape) == 4 and len(dwi1.shape) == 3:
            self.dict_output['main'] = self.inputs.dwi
            self.dict_output['secondary'] = self.inputs.dwi1
        elif len(dwi.shape) == 3 and len(dwi1.shape) == 4 and not topup:
            self.dict_output['main'] = self.inputs.dwi1
            self.dict_output['secondary'] = self.inputs.dwi
        elif len(dwi.shape) == 3 and len(dwi1.shape) == 3:
            self.dict_output['main'] = self.inputs.dwi
            self.dict_output['secondary'] = self.inputs.dwi1
        elif topup and len(dwi1.shape) == 4:
            ref = nib.load(self.inputs.dwi1)
            dwi1_b0 = dwi1[:, :, :, 0]
            im2save = nib.Nifti1Image(dwi1_b0, affine=ref.affine)
            nib.save(im2save, 'b0.nii.gz')
            self.dict_output['main'] = self.inputs.dwi
            self.dict_output['secondary'] = os.getcwd()+'/b0.nii.gz'

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()

        outputs["pe"] = self.dict_output['pe']
        outputs["main"] = self.dict_output['main']
        outputs["secondary"] = self.dict_output['secondary']
        if isdefined(self.dict_output['pe_1']):
            outputs["pe_1"] = self.dict_output['pe_1']

        return outputs


class CheckDwiNamesInputSpec(BaseInterfaceInputSpec):

    dicom_dwi = Directory()
    dicom_dwi1 = Directory()
    nifti_dwi = File()


class CheckDwiNamesOutputSpec(TraitedSpec):

    main = Directory()


class CheckDwiNames(BaseInterface):

    input_spec = CheckDwiNamesInputSpec
    output_spec = CheckDwiNamesOutputSpec

    def _run_interface(self, runtime):

        dwi = self.inputs.dicom_dwi
        dwi1 = self.inputs.dicom_dwi1
        nifti = self.inputs.nifti_dwi
        self.dict_output = {}

        if dwi.split('/')[-1] in nifti:
            self.dict_output['main'] = self.inputs.dicom_dwi
        elif dwi1.split('/')[-1] in nifti:
            self.dict_output['main'] = self.inputs.dicom_dwi1

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()

        outputs["main"] = self.dict_output['main']

        return outputs


class GenTopupConfigFilesInputSpec(BaseInterfaceInputSpec):

    ped = traits.Str(desc='phase encoding direction for the main image')


class GenTopupConfigFilesOutputSpec(TraitedSpec):

    config_file = File(exists=True, desc='configuration file for topup')
    apply_topup_config = File(
        exists=True, desc='configuration file for apply_topup')


class GenTopupConfigFiles(BaseInterface):

    input_spec = GenTopupConfigFilesInputSpec
    output_spec = GenTopupConfigFilesOutputSpec

    def _run_interface(self, runtime):

        ped = self.inputs.ped

        if ped == 'RL':
            with open('config_file.txt', 'w') as f:
                f.write('-1 0 0 1 \n1 0 0 1')
            f.close()
            with open('apply_topup_config_file.txt', 'w') as f:
                f.write('-1 0 0 1')
            f.close()
        elif ped == 'LR':
            with open('config_file.txt', 'w') as f:
                f.write('1 0 0 1 \n-1 0 0 1')
            f.close()
            with open('apply_topup_config_file.txt', 'w') as f:
                f.write('1 0 0 1')
            f.close()
        elif ped == 'AP':
            with open('config_file.txt', 'w') as f:
                f.write('0 -1 0 1 \n0 1 0 1')
            f.close()
            with open('apply_topup_config_file.txt', 'w') as f:
                f.write('0 -1 0 1')
            f.close()
        elif ped == 'AP':
            with open('config_file.txt', 'w') as f:
                f.write('0 1 0 1 \n0 -1 0 1')
            f.close()
            with open('apply_topup_config_file.txt', 'w') as f:
                f.write('0 1 0 1')
            f.close()

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()

        outputs["config_file"] = os.path.join(
            os.getcwd(), 'config_file.txt')
        outputs["apply_topup_config"] = os.path.join(
            os.getcwd(), 'apply_topup_config_file.txt')

        return outputs


class AffineMatrixGenerationInputSpec(BaseInterfaceInputSpec):

    motion_parameters = File(exists=True)
    reference_image = File(exists=True)


class AffineMatrixGenerationOutputSpec(TraitedSpec):

    affine_matrices = Directory(exists=True)


class AffineMatrixGeneration(BaseInterface):

    input_spec = AffineMatrixGenerationInputSpec
    output_spec = AffineMatrixGenerationOutputSpec

    def _run_interface(self, runtime):

        _, out_name, _ = split_filename(self.inputs.motion_parameters)
        motion_par = np.loadtxt(self.inputs.motion_parameters)
        motion_par = motion_par[:, :6]
        ref = nib.load(self.inputs.reference_image)
        ref_data = ref.get_data()
        # centre of mass
        if len(ref_data.shape) == 4:
            com = np.asarray(snm.center_of_mass(ref_data[:, :, :, 0]))
        else:
            com = np.asarray(snm.center_of_mass(ref_data))

        hdr = ref.header
        resolution = list(hdr.get_zooms()[:3])

        for i in range(len(motion_par)):
            mat = self.create_affine_mat(motion_par[i, :], resolution*com)
            np.savetxt(
                'affine_mat_{}.mat'.format(str(i).zfill(4)), mat, fmt='%f')

        affines = glob.glob('affine_mat*.mat')
        os.mkdir(out_name)
        for f in affines:
            shutil.move(f, out_name)

        return runtime

    def create_affine_mat(self, mp, cog):

        T = np.eye(4)

        T[0, -1] = cog[0]
        T[1, -1] = cog[1]
        T[2, -1] = cog[2]

        T_1 = np.linalg.inv(T)

        tx = mp[0]
        ty = mp[1]
        tz = mp[2]
        rx = mp[3]
        ry = mp[4]
        rz = mp[5]

        Rx = np.eye(3)
        Rx[1, 1] = np.cos(rx)
        Rx[1, 2] = np.sin(rx)
        Rx[2, 1] = -np.sin(rx)
        Rx[2, 2] = np.cos(rx)
        Ry = np.eye(3)
        Ry[0, 0] = np.cos(ry)
        Ry[0, 2] = -np.sin(ry)
        Ry[2, 0] = np.sin(ry)
        Ry[2, 2] = np.cos(ry)
        Rz = np.eye(3)
        Rz[0, 0] = np.cos(rz)
        Rz[0, 1] = np.sin(rz)
        Rz[1, 0] = -np.sin(rz)
        Rz[1, 1] = np.cos(rz)
        R_3x3 = np.dot(np.dot(Rx, Ry), Rz)

        m = np.eye(4)
        m[:3, :3] = R_3x3
        m[0, 3] = tx
        m[1, 3] = ty
        m[2, 3] = tz

        new_orig = np.dot(T, np.dot(m, T_1))[:, -1]

        m[0, 3] = new_orig[0]
        m[1, 3] = new_orig[1]
        m[2, 3] = new_orig[2]

        return m

    def _list_outputs(self):
        outputs = self._outputs().get()

        _, out_name, _ = split_filename(self.inputs.motion_parameters)

        outputs["affine_matrices"] = os.path.abspath(out_name)

        return outputs


class MeanDisplacementCalculationInputSpec(BaseInterfaceInputSpec):

    motion_mats = traits.List(desc='List of motion mats.')
    trs = traits.List(desc='List of repetition times.')
    start_times = traits.List(desc='List of start times.')
    real_durations = traits.List(desc='List of real durations.')
    reference = File(desc='Reference image.')
    input_names = traits.List(desc='List with the name of the inputs provided'
                              'by the user.')


class MeanDisplacementCalculationOutputSpec(TraitedSpec):

    mean_displacement = File(exists=True)
    mean_displacement_rc = File(exists=True)
    mean_displacement_consecutive = File(exists=True)
    start_times = File(exists=True)
    motion_parameters_rc = File(exists=True)
    motion_parameters = File(exists=True)
    offset_indexes = File(exists=True)
    mats4average = File(exists=True)
    corrupted_volumes = File(exists=True)


class MeanDisplacementCalculation(BaseInterface):

    input_spec = MeanDisplacementCalculationInputSpec
    output_spec = MeanDisplacementCalculationOutputSpec

    def _run_interface(self, runtime):

        list_inputs = zip(self.inputs.motion_mats, self.inputs.start_times,
                          self.inputs.real_durations, self.inputs.trs,
                          self.inputs.input_names)
        ref = nib.load(self.inputs.reference)
        ref_data = ref.get_data()
        # centre of gravity
        ref_cog = np.asarray(snm.center_of_mass(ref_data))
        list_inputs = sorted(list_inputs, key=lambda k: k[1])
        study_start_time = list_inputs[0][1]
        list_inputs = [
            (x[0], (dt.datetime.strptime(x[1], '%H%M%S.%f') -
                    dt.datetime.strptime(list_inputs[0][1], '%H%M%S.%f'))
             .total_seconds(), x[2], x[3], x[4]) for x in list_inputs]
        study_len = int((list_inputs[-1][1]+float(list_inputs[-1][2]))*1000)
        mean_displacement_rc = np.zeros(study_len)-1
        motion_par_rc = np.zeros((6, study_len))
        mean_displacement = []
        motion_par = []
        idt_mat = np.eye(4)
        all_mats = []
        all_mats4average = []
        start_times = []
        volume_names = []
        corrupted_volume_names = [
            'No unusual severe motion detected (i.e. no volume showed '
            'rotation greater than 8 degrees and/or translation greater '
            'than 20mm respect to the reference).']
        for f in list_inputs:
            mats = sorted(glob.glob(f[0]+'/*inv.mat'))
            mats4averge = sorted(glob.glob(f[0]+'/*mat.mat'))
            all_mats = all_mats+mats
            all_mats4average = all_mats4average+mats4averge
            start_scan = f[1]
            tr = f[3]
            if len(mats) > 1:
                for i, mat in enumerate(mats):
                    volume_names.append(f[-1]+'_vol_{}'
                                        .format(str(i+1).zfill(4)))
                    start_times.append((
                        dt.datetime.strptime(
                            study_start_time, '%H%M%S.%f') +
                        dt.timedelta(seconds=start_scan))
                                       .strftime('%H%M%S.%f'))
                    end_scan = start_scan+tr
                    m = np.loadtxt(mat)
                    md = self.rmsdiff(ref_cog, m, idt_mat)
                    mean_displacement_rc[
                        int(start_scan*1000):int(end_scan*1000)] = md
                    mean_displacement.append(md)
                    mp = self.avscale(m, ref_cog)
                    motion_par.append(mp)
                    duration = int(end_scan*1000)-int(start_scan*1000)
                    motion_par_rc[:, int(start_scan*1000):
                                  int(end_scan*1000)] = np.array(
                                      [mp, ]*duration).T
                    start_scan = end_scan
            elif len(mats) == 1:
                volume_names.append(f[-1])
                start_times.append((
                    dt.datetime.strptime(study_start_time,
                                         '%H%M%S.%f') +
                    dt.timedelta(seconds=start_scan)).strftime('%H%M%S.%f'))
                end_scan = start_scan+float(f[2])
                m = np.loadtxt(mats[0])
                md = self.rmsdiff(ref_cog, m, idt_mat)
                mean_displacement_rc[
                    int(start_scan*1000):int(end_scan*1000)] = md
                mean_displacement.append(md)
                mp = self.avscale(m, ref_cog)
                motion_par.append(mp)
                duration = int(end_scan*1000)-int(start_scan*1000)
                motion_par_rc[:, int(start_scan*1000):
                              int(end_scan*1000)] = np.array(
                                  [mp, ]*duration).T
        start_times.append((
            dt.datetime.strptime(study_start_time, '%H%M%S.%f') +
            dt.timedelta(seconds=end_scan)).strftime('%H%M%S.%f'))
        mean_displacement_consecutive = []
        for i in range(len(all_mats)-1):
            m1 = np.loadtxt(all_mats[i])
            m2 = np.loadtxt(all_mats[i+1])
            md_consecutive = self.rmsdiff(ref_cog, m1, m2)
            mean_displacement_consecutive.append(md_consecutive)

        corrupted_volumes = self.check_max_motion(motion_par)
        if corrupted_volumes:
            corrupted_volume_names = [
                'The following volumes showed an unusual severe motion (i.e. '
                'rotation greater than 8 degrees and/or translation greater '
                'than 20mm). \nThis is suspiciuos and can be potentially due '
                'to errors in the registration process. Please check the '
                'specified images before moving farward.\n']
            corrupted_volume_names = (
                corrupted_volume_names+[volume_names[x]
                                        for x in corrupted_volumes])
        offset_indexes = np.where(mean_displacement_rc == -1)
        for i in range(len(mean_displacement_rc)):
            if (mean_displacement_rc[i] == -1 and
                    mean_displacement_rc[i-1] != -1):
                mean_displacement_rc[i] = mean_displacement_rc[i-1]

        to_save = [mean_displacement, mean_displacement_consecutive,
                   mean_displacement_rc, motion_par_rc, start_times,
                   offset_indexes, all_mats4average, motion_par,
                   corrupted_volume_names]
        to_save_name = ['mean_displacement', 'mean_displacement_consecutive',
                        'mean_displacement_rc', 'motion_par_rc', 'start_times',
                        'offset_indexes', 'mats4average', 'motion_par',
                        'severe_motion_detection_report']
        for i in range(len(to_save)):
            np.savetxt(to_save_name[i]+'.txt', np.asarray(to_save[i]),
                       fmt='%s')

        return runtime

    def rmsdiff(self, cog, T1, T2):

        R = 80
        M = np.dot(T2, np.linalg.inv(T1))-np.identity(4)
        A = M[:3, :3]
        t = M[:3, 3]
        Tr = np.trace(np.dot(A.T, A))
        II = (t+np.dot(A, cog)).T
        III = t+np.dot(A, cog)
        cost = Tr*R**2/5
        rms = np.sqrt(cost + np.dot(II, III))

        return rms

    def avscale(self, mat, com, res=[1, 1, 1], moco=False):
        """Python implementation of the avscale function in fsl. However this
        works just with affine matrices from rigid body motion, i.e. it assumes
        that there is no scales or skew effect. Furthermore, if moco=True,
        it returns the rigid body motion parameters in Siemens moco series
        convetion."""
        c = np.asarray(com)
        trans_init = mat[:3, -1]
        rot_mat = mat[:3, :3]
        centre = c*res
        rot_x, rot_y, rot_z = self.rotationMatrixToEulerAngles(rot_mat)
        trans_tot = np.dot(rot_mat, centre)+trans_init-centre
        trans_x = trans_tot[0]
        trans_y = trans_tot[1]
        trans_z = trans_tot[2]
        if moco:
            rot_x_moco = -self.rad2degree(rot_y)
            rot_y_moco = self.rad2degree(rot_x)
            rot_z_moco = -self.rad2degree(rot_z)
            trans_x_moco = -trans_y
            trans_y_moco = trans_x
            trans_z_moco = -trans_z
            print [trans_x_moco, trans_y_moco, trans_z_moco, rot_x_moco,
                   rot_y_moco, rot_z_moco]
        return [rot_x, rot_y, rot_z, trans_x, trans_y, trans_z]

    def rad2degree(self, alpha_rad):
        return alpha_rad*180/np.pi

    def isRotationMatrix(self, R):
        Rt = np.transpose(R)
        shouldBeIdentity = np.dot(Rt, R)
        Identity = np.identity(3, dtype=R.dtype)
        n = np.linalg.norm(Identity - shouldBeIdentity)
        return n < 1e-4

    def rotationMatrixToEulerAngles(self, R):
        assert(self.isRotationMatrix(R))
        cy = math.sqrt(R[0, 0]*R[0, 0]+R[0, 1]*R[0, 1])
        singular = cy < 1e-4
        if not singular:
            cz = R[0, 0]/cy
            sz = R[0, 1]/cy
            cx = R[2, 2]/cy
            sx = R[1, 2]/cy
            sy = -R[0, 2]
            x = math.atan2(sx, cx)
            y = math.atan2(sy, cy)
            z = math.atan2(sz, cz)
        else:
            cx = R[1, 1]
            sx = -R[2, 1]
            sy = -R[0, 2]
            x = math.atan2(sx, cx)
            y = math.atan2(sy, 0.0)
            z = 0.0
        return np.array([x, y, z])

    def check_max_motion(self, motion_par):

        corrupted_vol_rot = np.where(np.abs(
            np.asarray(motion_par)[:, :3]) >= 0.14)[0]
        corrupted_vol_trans = np.where(np.abs(
            np.asarray(motion_par)[:, 3:]) >= 20)[0]
        corrupted_vol = list(set(corrupted_vol_rot.tolist() +
                                 corrupted_vol_trans.tolist()))

        return corrupted_vol

    def _list_outputs(self):
        outputs = self._outputs().get()

        outputs["mean_displacement"] = os.getcwd()+'/mean_displacement.txt'
        outputs["mean_displacement_rc"] = (
            os.getcwd()+'/mean_displacement_rc.txt')
        outputs["mean_displacement_consecutive"] = (
            os.getcwd()+'/mean_displacement_consecutive.txt')
        outputs["start_times"] = os.getcwd()+'/start_times.txt'
        outputs["motion_parameters"] = os.getcwd()+'/motion_par.txt'
        outputs["motion_parameters_rc"] = os.getcwd()+'/motion_par_rc.txt'
        outputs["offset_indexes"] = os.getcwd()+'/offset_indexes.txt'
        outputs["mats4average"] = os.getcwd()+'/mats4average.txt'
        outputs["corrupted_volumes"] = (
            os.getcwd()+'/severe_motion_detection_report.txt')

        return outputs


class MotionFramingInputSpec(BaseInterfaceInputSpec):

    mean_displacement = File(exists=True)
    mean_displacement_consec = File(exists=True)
    start_times = File(exists=True)
    motion_threshold = traits.Float(desc='Everytime the mean displacement is '
                                    'greater than this value (in mm), a new '
                                    'frame will be initialised. Default 2mm',
                                    default=2)
    temporal_threshold = traits.Float(desc='If one frame temporal duration is '
                                      'shorter than this value (in sec) then '
                                      'the frame will be discarded. Default '
                                      '30sec', default=30)
#    pet_data_dir = Directory(
#        exists=True, desc='PET directory with all the acquired data from the '
#        'scanner. This folder must contain at list the list-mode data and its'
#        'headerto which will be used to extract the pet start time and end '
#        'time. This must be provided if motion detection has to be then '
#        'applied to PET motion correction.', default=None)
    pet_start_time = traits.Str(desc='PET start time', default=None)
    pet_end_time = traits.Str(desc='PET end time', default=None)


class MotionFramingOutputSpec(TraitedSpec):

    frame_start_times = File(exists=True)
    frame_vol_numbers = File(exists=True, desc='Text file with the number of '
                             'volume where the motion occurred.')
    timestamps_dir = Directory(desc='Directory with the timestamps for all'
                               ' the detected frames')


class MotionFraming(BaseInterface):

    input_spec = MotionFramingInputSpec
    output_spec = MotionFramingOutputSpec

    def _run_interface(self, runtime):

        mean_displacement = np.loadtxt(self.inputs.mean_displacement,
                                       dtype=float)
        mean_displacement_consecutive = np.loadtxt(
            self.inputs.mean_displacement_consec, dtype=float)
        th = self.inputs.motion_threshold
        start_times = np.loadtxt(self.inputs.start_times, dtype=str)
        temporal_th = self.inputs.temporal_threshold
        pet_st = self.inputs.pet_start_time
        pet_endtime = self.inputs.pet_end_time
        if not pet_st and not pet_endtime:
            pet_st = None
            pet_endtime = None

        md_0 = mean_displacement[0]
        max_md = mean_displacement[0]
        frame_vol = [0]
        frame_st4pet = []

        scan_duration = [
            (dt.datetime.strptime(start_times[i], '%H%M%S.%f') -
             dt.datetime.strptime(start_times[i-1], '%H%M%S.%f')
             ).total_seconds() for i in range(1, len(start_times))]

        for i, md in enumerate(mean_displacement[1:]):

            current_md = md
            if (np.abs(md_0 - current_md) > th or
                    np.abs(max_md - current_md) > th):
                duration = np.sum(scan_duration[frame_vol[-1]:i+1])
                if duration > temporal_th:
                    if i+1 not in frame_vol:
                        frame_vol.append(i+1)

                    md_0 = current_md
                    max_md = current_md
                else:
                    prev_md = mean_displacement[frame_vol[-1]]
                    if (prev_md - current_md) > th*2:
                        frame_vol.remove(frame_vol[-1])
                    elif (current_md - prev_md) > th:
                        frame_vol.remove(frame_vol[-1])
                        frame_vol.append(i)
            elif mean_displacement_consecutive[i] > th:
                duration = np.sum(scan_duration[frame_vol[-1]:i+1])
                if duration > temporal_th:
                    if i+1 not in frame_vol:
                        frame_vol.append(i+1)
                    md_0 = current_md
                    max_md = current_md
            elif current_md > max_md:
                max_md = current_md
            elif current_md < md_0:
                md_0 = current_md

        duration = np.sum(scan_duration[frame_vol[-1]:i+2])
        if duration > temporal_th:
            if (i + 2) not in frame_vol:
                frame_vol.append(i + 2)
        else:
            frame_vol.remove(frame_vol[-1])
            frame_vol.append(i + 2)

        frame_vol = sorted(frame_vol)
        frame_start_times = [start_times[x] for x in frame_vol]
        if pet_st and pet_endtime:
            frame_st4pet = [
                x for x in frame_start_times if
                (dt.datetime.strptime(x, '%H%M%S.%f') >
                 dt.datetime.strptime(pet_st, '%H%M%S.%f')
                 and dt.datetime.strptime(x, '%H%M%S.%f') <
                 dt.datetime.strptime(pet_endtime, '%H%M%S.%f'))]
            if frame_start_times[0] in frame_st4pet:
                frame_st4pet.remove(frame_start_times[0])
            if frame_start_times[-1] in frame_st4pet:
                frame_st4pet.remove(frame_start_times[-1])
            frame_st4pet.append(pet_st)
            frame_st4pet.append(pet_endtime)
            frame_st4pet = sorted(frame_st4pet)
            if ((dt.datetime.strptime(frame_st4pet[1], '%H%M%S.%f') -
                    dt.datetime.strptime(frame_st4pet[0], '%H%M%S.%f'))
                    .total_seconds() < 30):
                frame_st4pet.remove(frame_st4pet[1])
            if ((dt.datetime.strptime(frame_st4pet[-1], '%H%M%S.%f') -
                    dt.datetime.strptime(frame_st4pet[-2], '%H%M%S.%f'))
                    .total_seconds() < 30):
                frame_st4pet.remove(frame_st4pet[-2])
            frame_vol = [i for i in range(len(start_times)) for j in
                         range(len(frame_st4pet)) if start_times[i] ==
                         frame_st4pet[j]]
            if (dt.datetime.strptime(start_times[0], '%H%M%S.%f') >
                    dt.datetime.strptime(pet_st, '%H%M%S.%f')):
                frame_vol.append(0)
            else:
                vol = [i for i in range(len(start_times)) if
                       (dt.datetime.strptime(start_times[i], '%H%M%S.%f') <
                        dt.datetime.strptime(frame_st4pet[0], '%H%M%S.%f') and
                        dt.datetime.strptime(start_times[i+1], '%H%M%S.%f') >
                        dt.datetime.strptime(frame_st4pet[0], '%H%M%S.%f'))]
                frame_vol.append(vol[0])
            if (dt.datetime.strptime(start_times[-1], '%H%M%S.%f') <
                    dt.datetime.strptime(pet_endtime, '%H%M%S.%f')):
                frame_vol.append(len(start_times)-1)
            else:
                vol = [i for i in range(len(start_times)) if
                       (dt.datetime.strptime(start_times[i], '%H%M%S.%f') <
                        dt.datetime.strptime(frame_st4pet[-1], '%H%M%S.%f') and
                        dt.datetime.strptime(start_times[i+1], '%H%M%S.%f') >
                        dt.datetime.strptime(frame_st4pet[-1], '%H%M%S.%f'))]
                frame_vol.append(vol[0])
            frame_vol = sorted(frame_vol)
        np.savetxt('frame_start_times.txt', np.asarray(frame_start_times),
                   fmt='%s')
        os.mkdir('timestamps')
        if frame_st4pet:
            timestamps_2save = frame_st4pet
        else:
            timestamps_2save = frame_start_times
        np.savetxt('timestamps/frame_start_times_4PET.txt',
                   np.asarray(timestamps_2save), fmt='%s')
        for i in range(len(timestamps_2save)-1):
            with open('timestamps/timestamps_Frame{}.txt'
                      .format(str(i).zfill(3)), 'w') as f:
                f.write(timestamps_2save[i]+'\n'+timestamps_2save[i+1])
            f.close()
        np.savetxt('frame_vol_numbers.txt', np.asarray(frame_vol), fmt='%s')

        return runtime

    def pet_time_info(self, pet_data_dir):

        pet_duration = None
        for root, dirs, files in os.walk(pet_data_dir):
            bf_files = [f for f in files if not f[0] == '.' and '.bf' in f]
            dirs[:] = [d for d in dirs if not d[0] == '.']

#         bf_files = glob.glob('{}/*.bf'.format(pet_data_dir))
        if not bf_files:
            pet_start_time = None
            pet_endtime = None
            print ('No .bf file found in {}. If you want to perform motion '
                   'correction please provide the right pet data. ')
        else:
            max_size = 0
            for bf in bf_files:
                size = os.path.getsize(os.path.join(root, bf))
                if size > max_size:
                    max_size = size
                    list_mode_file = os.path.join(root, bf)

            pet_image = list_mode_file.split('.bf')[0] + '.dcm'
            try:
                hd = pydicom.read_file(pet_image)
                pet_start_time = hd.AcquisitionTime
            except AttributeError:
                pet_start_time = None
            with open(pet_image, 'r') as f:
                for line in f:
                    if 'image duration' in line:
                        pet_duration = line.strip()
                        pet_duration = int(pet_duration.split(':=')[-1])
            if pet_duration:
                pet_endtime = ((
                    dt.datetime.strptime(pet_start_time, '%H%M%S.%f') +
                    dt.timedelta(seconds=pet_duration))
                                    .strftime('%H%M%S.%f'))
                pet_duration = pet_duration
            else:
                pet_endtime = None

        return pet_start_time, pet_endtime

    def _list_outputs(self):
        outputs = self._outputs().get()

        outputs["frame_start_times"] = os.getcwd()+'/frame_start_times.txt'
        outputs["frame_vol_numbers"] = os.getcwd()+'/frame_vol_numbers.txt'
        outputs["timestamps_dir"] = os.getcwd()+'/timestamps'

        return outputs


class PlotMeanDisplacementRCInputSpec(BaseInterfaceInputSpec):

    mean_disp_rc = File(exists=True, desc='Text file containing the mean '
                        'displacement real clock.')
    frame_start_times = File(exists=True, desc='Frame start times as detected'
                             'by the motion framing pipeline')
    false_indexes = File(exists=True, desc='Time indexes were the scanner was '
                         'idling, i.e. there is no motion information.')
    framing = traits.Bool(desc='If true, the frame start times will be plotted'
                          'in the final image.')


class PlotMeanDisplacementRCOutputSpec(TraitedSpec):

    mean_disp_plot = File(exists=True, desc='Mean displacement plot.')


class PlotMeanDisplacementRC(BaseInterface):

    input_spec = PlotMeanDisplacementRCInputSpec
    output_spec = PlotMeanDisplacementRCOutputSpec

    def _run_interface(self, runtime):

        mean_disp_rc = np.loadtxt(self.inputs.mean_disp_rc)
        frame_start_times = np.loadtxt(self.inputs.frame_start_times)
        false_indexes = np.loadtxt(self.inputs.false_indexes, dtype=int)
        framing = self.inputs.framing
        plot_offset = True
        dates = np.arange(0, len(mean_disp_rc), 1)
        indxs = np.zeros(len(mean_disp_rc), int)+1
        indxs[false_indexes] = 0
        start_true_period = [x for x in range(1, len(indxs)) if indxs[x] == 1
                             and indxs[x-1] == 0]
        end_true_period = [x for x in range(len(indxs)-1) if indxs[x] == 0 and
                           indxs[x-1] == 1]
        start_true_period.append(1)
        end_true_period.append(len(dates))
        start_true_period = sorted(start_true_period)
        end_true_period = sorted(end_true_period)
        if len(start_true_period) == len(end_true_period)-1:
            end_true_period.remove(end_true_period[-1])
        elif len(start_true_period) != len(end_true_period):
            print ('Something went wrong in the indentification of the MR '
                   'idling time. It will not be plotted.')
            plot_offset = False
#         true_indexes = dates[indxs == 1]
#         true_indexes = [x for x in dates if x not in false_indexes]

        fig, ax = plot.subplots()
        fig.set_size_inches(21, 9)
        font = {'weight': 'bold', 'size': 30}
        matplotlib.rc('font', **font)
        ax.set_xlim(0, dates[-1])
        ax.set_ylim(-0.3, np.max(mean_disp_rc) + 1)
        if plot_offset:
            for i in range(0, len(start_true_period)):
                ax.plot(dates[start_true_period[i]-1:end_true_period[i]+1],
                        mean_disp_rc[start_true_period[i]-1:
                                     end_true_period[i]+1],
                        c='b', linewidth=2)
            for i in range(0, len(end_true_period)-1):
                ax.plot(
                    dates[end_true_period[i]-1:start_true_period[i+1]+1],
                    mean_disp_rc[end_true_period[i]-1:
                                 start_true_period[i+1]+1],
                    c='b', linewidth=2, ls='--', dashes=(2, 3))

        if framing:
            cl = 'yellow'
            for i in range(len(frame_start_times[:-1])):

                tt = (
                    (dt.datetime.strptime(str(frame_start_times[i]),
                                          '%H%M%S.%f') -
                     dt.datetime.strptime(str(frame_start_times[0]),
                                          '%H%M%S.%f'))
                    .total_seconds()*1000)
                if tt >= len(dates):
                    tt = len(dates)-1
                plot.axvline(dates[int(tt)], c='b', alpha=0.3, ls='--')

                tt1 = ((dt.datetime.strptime(str(frame_start_times[i+1]),
                                             '%H%M%S.%f') -
                       dt.datetime.strptime(str(frame_start_times[0]),
                                            '%H%M%S.%f'))
                       .total_seconds()*1000)
                if tt1 >= len(dates):
                    tt1 = len(dates)-1
                plot.axvspan(dates[int(tt)], dates[int(tt1)], facecolor=cl,
                             alpha=0.4, linewidth=0)

                if i % 2 == 0:
                    cl = 'w'
                else:
                    cl = 'yellow'

        indx = np.arange(0, len(dates), 300000)
        my_thick = [str(i) for i in np.arange(0, len(dates)/60000, 5)]
        plot.xticks(dates[indx], my_thick)

        plot.savefig('mean_displacement_real_clock.png')
        plot.close()

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()

        outputs["mean_disp_plot"] = (
            os.getcwd()+'/mean_displacement_real_clock.png')

        return outputs


class AffineMatAveragingInputSpec(BaseInterfaceInputSpec):

    frame_vol_numbers = File(exists=True)
    all_mats4average = File(exists=True)


class AffineMatAveragingOutputSpec(TraitedSpec):

    average_mats = Directory(exists=True, desc='directory with all the average'
                             ' transformation matrices for each detected '
                             'frame.')


class AffineMatAveraging(BaseInterface):

    input_spec = AffineMatAveragingInputSpec
    output_spec = AffineMatAveragingOutputSpec

    def _run_interface(self, runtime):

        frame_vol = np.loadtxt(self.inputs.frame_vol_numbers, dtype=int)
        all_mats = np.loadtxt(self.inputs.all_mats4average, dtype=str)
        idt = np.eye(4)

        for v in range(len(frame_vol)-1):

            v1 = frame_vol[v]
            v2 = frame_vol[v + 1]
            mat_tot = np.zeros((4, 4, (v2 - v1)))
            n_vol = 0

            for j, m in enumerate(all_mats[v1:v2]):
                mat = np.loadtxt(m)
                if (mat == idt).all():
                    mat_tot[:, :, j] = np.zeros((4, 4))
                else:
                    mat_tot[:, :, j] = mat
                    n_vol += 1
            if n_vol > 0:
                average_mat = np.sum(mat_tot, axis=2) / n_vol
            else:
                average_mat = idt

            np.savetxt(
                'average_matrix_vol_{0}-{1}.txt'
                .format(str(v1).zfill(4), str(v2).zfill(4)), average_mat)

        if os.path.isdir('frame_mean_transformation_mats') is False:
            os.mkdir('frame_mean_transformation_mats')

        mats = glob.glob('average_matrix_vol*')
        for m in mats:
            shutil.move(m, 'frame_mean_transformation_mats')

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()

        outputs["average_mats"] = (
            os.getcwd()+'/frame_mean_transformation_mats')

        return outputs


class PetCorrectionFactorInputSpec(BaseInterfaceInputSpec):

    timestamps = Directory(exists=True, desc='Frame start times as detected'
                           'by the motion framing pipeline')


class PetCorrectionFactorOutputSpec(TraitedSpec):

    corr_factors = File(exists=True, desc='Pet correction factors.')


class PetCorrectionFactor(BaseInterface):

    input_spec = PetCorrectionFactorInputSpec
    output_spec = PetCorrectionFactorOutputSpec

    def _run_interface(self, runtime):

        frame_st = np.loadtxt(
            self.inputs.timestamps+'/frame_start_times_4PET.txt', dtype=str)
        start = dt.datetime.strptime(frame_st[0], '%H%M%S.%f')
        end = dt.datetime.strptime(frame_st[-1], '%H%M%S.%f')
        tot_duration = (end - start).total_seconds()
        corr_factors = []
        for t in range(len(frame_st) - 1):
            start = dt.datetime.strptime(frame_st[t], '%H%M%S.%f')
            end = dt.datetime.strptime(frame_st[t + 1], '%H%M%S.%f')
            d = (end - start).total_seconds()
            corr_factors.append(d / tot_duration)

        with open('correction_factors_PET_data.txt', 'w') as f:
            for el in corr_factors:
                f.write(str(el) + '\n')
            f.close()

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()

        outputs["corr_factors"] = (
            os.getcwd()+'/correction_factors_PET_data.txt')

        return outputs


class FrameAlign2ReferenceInputSpec(BaseInterfaceInputSpec):

    average_mats = Directory(exists=True, desc='directory with all the average'
                             ' transformation matrices for each detected '
                             'frame.')
    ute_regmat = File(exists=True, desc='registration mat between ute image '
                      'and reference.')
    ute_qform_mat = File(exists=True, desc='qform mat between ute and '
                         'reference.')
    fixed_binning = traits.Bool(desc='if true, the function will assume that '
                                'the average matrices have been generated for '
                                'dynamic motion correction. Default is False.')
    umap = File(exists=True, desc='If a umap is provided, the function will '
                'align it to the head position in each frame. Default is None',
                default=None)
    pct = traits.Bool(desc='if True, the function will assume that the '
                      'provided umap is continuos values, as the pseudo CT '
                      'umap. Otherwise, it will assume that the values are '
                      'discrete. Default is False.')


class FrameAlign2ReferenceOutputSpec(TraitedSpec):

    frame2reference_mats = Directory(exists=True, desc='directory with all the'
                                     ' matrices which align each frame to the '
                                     'reference.')
    umaps_align2ref = Directory(desc='directory with all the realigned umaps '
                                '(if a umaps is provided as input).')


class FrameAlign2Reference(BaseInterface):

    input_spec = FrameAlign2ReferenceInputSpec
    output_spec = FrameAlign2ReferenceOutputSpec

    def _run_interface(self, runtime):

        fixed_binning = self.inputs.fixed_binning
        average_mats = sorted(glob.glob(self.inputs.average_mats+'/*.txt'))
        umap = self.inputs.umap
        pct = self.inputs.pct
        ute_regmat = self.inputs.ute_regmat
        ute_qform_mat = self.inputs.ute_qform_mat
        if not fixed_binning:
            outname = 'Frame'
        else:
            outname = 'Bin'

        for i, mat in enumerate(average_mats):
            self.FrameAlign2Reference_calc(mat, i, ute_regmat, ute_qform_mat,
                                           outname, umap=umap, pct=pct)

        if os.path.isdir('{}_align2ref_mats'.format(outname)) is False:
            os.mkdir('{}_align2ref_mats'.format(outname))

        mats = glob.glob('*ref_to_ute*.mat')
        for m in mats:
            shutil.move(m, '{}_align2ref_mats'.format(outname))
        if umap:
            umaps = glob.glob('Frame_*_umap.nii.gz')
            if os.path.isdir('umaps_align2ref') is False:
                os.mkdir('umaps_align2ref')
            for u in umaps:
                shutil.move(u, 'umaps_align2ref')

        return runtime

    def FrameAlign2Reference_calc(self, mat, i, ute_regmat, ute_qform_mat,
                                  outname, umap=None, pct=False):

        mat = np.loadtxt(mat)
        utemat = np.loadtxt(ute_regmat)
        utemat_qform = np.loadtxt(ute_qform_mat)
        utemat_qform_inv = np.linalg.inv(utemat_qform)
        ute2frame = np.dot(mat, utemat)
        ute2frame_qform = np.dot(utemat_qform_inv, ute2frame)
        ute2frame_qform_inv = np.linalg.inv(ute2frame_qform)

        np.savetxt('{0}_{1}_ref_to_ute.mat'.format(outname, str(i).zfill(3)),
                   ute2frame_qform)
        np.savetxt(
            '{0}_{1}_ref_to_ute_inv.mat'.format(outname, str(i).zfill(3)),
            ute2frame_qform_inv)

        if umap:
            if pct:
                interp = 'trilinear'
            else:
                interp = 'nearestneighbour'
            flt = fsl.FLIRT(bins=256, cost_func='corratio')
            flt.inputs.reference = umap
            flt.inputs.in_file = umap
            flt.inputs.interp = interp
            flt.inputs.in_matrix_file = ('Frame_{0}_ref_to_ute.mat'
                                         .format(str(i).zfill(3)))
            flt.inputs.out_file = 'Frame_{0}_umap.nii.gz'.format(
                str(i).zfill(3))
            flt.inputs.apply_xfm = True
            flt.run()

    def _list_outputs(self):
        outputs = self._outputs().get()
        fixed_binning = self.inputs.fixed_binning
        if not fixed_binning:
            outname = 'Frame'
        else:
            outname = 'Bin'

        outputs["frame2reference_mats"] = (
            os.getcwd()+'/{}_align2ref_mats'.format(outname))
        if self.inputs.umap:
            outputs["umaps_align2ref"] = (
                os.getcwd()+'/umaps_align2ref')

        return outputs


class CreateMocoSeriesInputSpec(BaseInterfaceInputSpec):

    moco_template = File(exists=True, mandatory=True, desc='Existing moco '
                         'series template to modify according to the extracted'
                         'motion information.')
    motion_par = File(exists=True, mandatory=True, desc='Text file with the '
                      'motion parameters extracted by the mean displacement '
                      'calculation pipeline.')
    start_times = File(exists=True, mandatory=True, desc='start times of all '
                       'the sequences (or volumes) acquired in the study ('
                       'this is the output of the mean displacement calculatio'
                       'n pipeline).')


class CreateMocoSeriesOutputSpec(TraitedSpec):

    modified_moco = Directory(entries=True, desc='Directory with the new moco '
                              'series')


class CreateMocoSeries(BaseInterface):

    input_spec = CreateMocoSeriesInputSpec
    output_spec = CreateMocoSeriesOutputSpec

    def _run_interface(self, runtime):

        moco_template = self.inputs.moco_template
        motion_par = np.loadtxt(self.inputs.motion_par)
        start_times = np.loadtxt(self.inputs.start_times, dtype=str)
        start_times = start_times[:-1]

        if len(motion_par) != len(start_times):
            raise Exception('Detected a different number of motion parameters '
                            'and start times. This number must be the same in '
                            'order to create a new moco series. Please check.')
        motion_par_moco = [self.fsl2moco(x) for x in motion_par]
        new_uid = pydicom.uid.generate_uid()
        for i in range(len(start_times)):
            hd = pydicom.read_file(moco_template)
            for n in range(3):
                hd[0x19, 0x1025].value[n] = motion_par_moco[i][n]
            for n in range(3):
                hd[0x19, 0x1026].value[n] = motion_par_moco[i][n+3]
            hd.AcquisitionTime = start_times[i]
            hd.InstanceNumber = pydicom.valuerep.IS(i+1)
            hd.AcquisitionNumber = pydicom.valuerep.IS(i+1)
            hd.SeriesInstanceUID = new_uid
            hd.SeriesDescription = 'MoCoSeries'
            hd.SeriesNumber = '150'
            hd.save_as('{}.IMA'.format(str(i).zfill(6)))

        os.mkdir('new_moco_series')
        dicoms = sorted(glob.glob('*.IMA'))
        for dcm in dicoms:
            shutil.move(dcm, 'new_moco_series')

        return runtime

    def fsl2moco(self, mp):
        rot_x_moco = -self.rad2degree(mp[1])
        rot_y_moco = self.rad2degree(mp[0])
        rot_z_moco = -self.rad2degree(mp[2])
        trans_x_moco = -mp[4]
        trans_y_moco = mp[3]
        trans_z_moco = -mp[5]
        return [trans_x_moco, trans_y_moco, trans_z_moco, rot_x_moco,
                rot_y_moco, rot_z_moco]

    def rad2degree(self, alpha_rad):
        return alpha_rad*180/np.pi

    def _list_outputs(self):
        outputs = self._outputs().get()

        outputs["modified_moco"] = os.getcwd()+'/new_moco_series'

        return outputs


class FixedBinningInputSpec(BaseInterfaceInputSpec):

    n_frames = traits.Int(desc='Number of frames you want to have '
                          'realignment matrices for.')
    pet_offset = traits.Int(desc='seconds from the start of the PET you want '
                            'to discard before starting the data binning.')
    bin_len = traits.Int(desc='Temporal length in seconds for each bin.')
    start_times = File(desc='Start times of all the scans in the study. This '
                       'is generated by mean displacement calculation '
                       'pipeline.')
    pet_duration = traits.Int(desc='PET temporal duration in seconds.')
    pet_start_time = traits.Str(desc='PET start time')
    motion_mats = File(exists=True, desc='Text file with the list of all the '
                       'motion matrices.')


class FixedBinningOutputSpec(TraitedSpec):

    average_bin_mats = Directory(desc='Directory with all the matrices to be '
                                 'used to realign the reconstructed '
                                 'fixed-binning PET images.')


class FixedBinning(BaseInterface):
    """
        Function to divide PET list-mode files into a fixed number of bins
        with specified length.
        Motion matrices for each bin are then calculated averaging all
        the matrices within that bin.

        Parameters
        ----------
        mc : Boolean
            If True, motion correction within bins is performed. If bin
            length is less than 2 minutes, the motion correction cannot be
            performed.
        bin_len : Integer
            Temporal length of each PET bin.
        pet_offset : Integer
            Time offset (in seconds) to start the fixed binning from a
            time point different from the PET start time. Default is zero.
        n_frames : Integer
            Number of PET frames you want to correct for motion. By default
            is the total lenght of the PET acquisition divided by the
            bin_len.

        Returns
        -------
        One motion matrix and one timestamp per bin. If mc=True it will
        return motion matrices and timestamps for each sub-bin too.
    """

    input_spec = FixedBinningInputSpec
    output_spec = FixedBinningOutputSpec

    def _run_interface(self, runtime):

        n_frames = self.inputs.n_frames
        pet_offset = self.inputs.pet_offset
        bin_len = self.inputs.bin_len
        start_times = np.loadtxt(self.inputs.start_times, dtype=str)
        pet_duration = self.inputs.pet_duration
        pet_start_time = self.inputs.pet_start_time
        motion_mats = np.loadtxt(self.inputs.motion_mats, dtype=str)
        if n_frames == 0 and pet_offset == 0:
            pet_len = pet_duration
        elif n_frames == 0 and pet_offset != 0:
            pet_len = pet_duration-pet_offset
        elif n_frames != 0:
            pet_len = bin_len*n_frames

        MR_start_time = dt.datetime.strptime(str(start_times[0]), '%H%M%S.%f')
        start_times_diff = [
            (dt.datetime.strptime(str(start_times[i+1]), '%H%M%S.%f') -
             dt.datetime.strptime(
                 str(start_times[i]), '%H%M%S.%f')).total_seconds()
            for i in range(len(start_times)-1)]
        scan_duration = np.cumsum(np.asarray(start_times_diff))

        pet_st = (dt.datetime.strptime(pet_start_time, '%H%M%S.%f') +
                  dt.timedelta(seconds=pet_offset))
        PetBins = [pet_st+dt.timedelta(seconds=x) for x in
                   range(0, pet_len, bin_len)]
        MrBins = [MR_start_time+dt.timedelta(seconds=x)
                  for x in scan_duration]
        MrStartPoints = [MrBins[i]+dt.timedelta(
            seconds=(MrBins[i+1]-MrBins[i]).total_seconds()/2) for i in
                         range(len(MrBins)-1)]

        indxs = []
        PetBins.append(pet_st+dt.timedelta(seconds=pet_len))
        if pet_offset != 0:
            print ('PET start time offset of {0} seconds detected. '
                   'Fixed binning will start at {2} and will last '
                   'for {1} seconds.'.format(str(pet_offset), str(pet_len),
                                             pet_st.strftime('%H%M%S.%f')))
        for pet_bin in PetBins:

            for i in range(len(MrStartPoints)-1):
                if (pet_bin > MrStartPoints[i] and
                        pet_bin < MrStartPoints[i+1]):
                    MrDiff = (
                        (MrStartPoints[i+1]-MrStartPoints[i]).total_seconds())
                    w0 = (MrStartPoints[i+1]-pet_bin).total_seconds()/MrDiff
                    w1 = (pet_bin-MrStartPoints[i]).total_seconds()/MrDiff
                    indxs.append([[w0, i], [w1, i+1]])
                    break
                elif pet_bin < MrStartPoints[i]:
                    indxs.append([[1, i], [0, i+1]])
                    break
        while len(indxs) < len(PetBins):
            indxs.append([[0, len(MrStartPoints)-2],
                          [1, len(MrStartPoints)-1]])
        z = 0
        for ii in range(len(indxs)-1):
            start = indxs[ii]
            end = indxs[ii+1]
            s1 = start[0][1]
            e1 = start[1][1]
            s2 = end[0][1]
            e2 = end[1][1]
            if s1 == s2 and e1 == e2:
                mat_s1 = np.loadtxt(motion_mats[s1])
                mat_e1 = np.loadtxt(motion_mats[e1])
                av_mat = start[0][0]*mat_s1 + start[1][0]*mat_e1
                np.savetxt(
                    'average_motion_mat_bin_{0}.txt'.format(str(z).zfill(3)),
                    av_mat)
                z = z+1
            elif (s1+1 == s2 and e1+1 == e2) or (s1+2 == s2 and e1+2 == e2):
                mat_s1 = np.loadtxt(motion_mats[s1])
                mat_e1 = np.loadtxt(motion_mats[e1])
                mat_s2 = np.loadtxt(motion_mats[s2])
                mat_e2 = np.loadtxt(motion_mats[e2])
                av_mat_1 = start[0][0]*mat_s1 + start[1][0]*mat_e1
                av_mat_2 = end[0][0]*mat_s2 + end[1][0]*mat_e2
                mean_mat = (av_mat_1 + av_mat_2)/2
                np.savetxt(
                    'average_motion_mat_bin_{0}.txt'.format(str(z).zfill(3)),
                    mean_mat)
                z = z+1
            else:
                mat_tot = np.zeros((4, 4, (s2-s1)))
                mat_s1 = np.loadtxt(motion_mats[s1])
                mat_e1 = np.loadtxt(motion_mats[e1])
                mat_s2 = np.loadtxt(motion_mats[s2])
                mat_e2 = np.loadtxt(motion_mats[e2])
    #                     m1 = start[0][0]*mat_s1 + start[1][0]*mat_e1
                mat_tot[:, :, 0] = (
                    start[0][0]*mat_s1 + start[1][0]*mat_e1)
                mat_tot[:, :, -1] = (
                    end[0][0]*mat_s2 + end[1][0]*mat_e2)
                for i, m in enumerate(range(e1+1, s2)):
                    mat_tot[:, :, i+1] = np.loadtxt(motion_mats[m])
                mean_mat = np.mean(mat_tot, axis=2)
                np.savetxt(
                    'average_motion_mat_bin_{0}.txt'
                    .format(str(z).zfill(3)), mean_mat)
                z = z+1
        os.mkdir('average_bin_mats')
        files = glob.glob('*bin*.txt')
        for f in files:
            shutil.move(f, 'average_bin_mats')

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()

        outputs["average_bin_mats"] = os.getcwd()+'/average_bin_mats'

        return outputs
