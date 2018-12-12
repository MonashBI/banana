

from nipype.interfaces.base import (BaseInterface, BaseInterfaceInputSpec,
                                    traits, TraitedSpec, Directory, File,
                                    isdefined)
import numpy as np
import glob
import pydicom
from nipype.utils.filemanip import split_filename
import datetime as dt
import os.path
import nibabel as nib
from arcana.utils import split_extension
import nibabel.nicom.csareader as csareader
from logging import getLogger
from banana.exceptions import BananaMissingHeaderValue


logger = getLogger('banana')


PEDP_TO_SIGN = {0: '-1', 1: '+1'}


class DicomHeaderInfoExtractionInputSpec(BaseInterfaceInputSpec):

    dicom_folder = Directory(exists=True, desc='Directory with DICOM files',
                             mandatory=True)
    multivol = traits.Bool(desc='Specify whether a scan is 3D or 4D',
                           default=False)
    reference = traits.Bool(desc='Specify whether the input scan is the motion'
                            ' correction reference.')


class DicomHeaderInfoExtractionOutputSpec(TraitedSpec):

    tr = traits.Float(desc='Repetition time.')
    echo_times = traits.List(traits.Float(), desc='Echo times')
    voxel_sizes = traits.List(traits.Float(), desc="Voxel sizes")
    H = traits.List((traits.Float(), traits.Float(), traits.Float),
                    desc="Main magnetic field ")
    B0 = traits.Float(desc="Main magnetic field strength")
    start_time = traits.Float(desc='Scan start time.')
    real_duration = traits.Float(
        desc=('For 4D files, this will be the number of '
              'volumes multiplied by the TR.'))
    total_duration = traits.Float(
        desc='Scan duration as extracted from the header.')
    ped = traits.Str(desc='Phase encoding direction.')
    pe_angle = traits.Str(desc='Phase angle.')
    dcm_info = File(exists=True, desc='File with all the previous outputs.')
    ref_motion_mats = Directory(desc='folder with the reference motion mats')


class DicomHeaderInfoExtraction(BaseInterface):

    input_spec = DicomHeaderInfoExtractionInputSpec
    output_spec = DicomHeaderInfoExtractionOutputSpec

    def _run_interface(self, runtime):

        list_dicom = sorted(glob.glob(self.inputs.dicom_folder + '/*'))
        multivol = self.inputs.multivol
        _, out_name, _ = split_filename(self.inputs.dicom_folder)
        ped = ''
        phase_offset = ''
        self.dict_output = {}
        dwi_directions = None

        try:
            phase_offset, ped = self.get_phase_encoding_direction(
                list_dicom[0])
        except KeyError:
            pass  # image does not have ped info in the header

        with open(list_dicom[0], 'rb') as f:
            for line in f:
                try:
                    line = line[:-1].decode('utf-8')
                except UnicodeDecodeError:
                    continue
                if 'TotalScan' in line:
                    total_duration = line.split('=')[-1].strip()
                    if not multivol:
                        real_duration = total_duration
                elif 'alTR[0]' in line:
                    tr = float(line.split('=')[-1].strip()) / 1000000
                elif ('SliceArray.asSlice[0].dInPlaneRot' in line and
                        (not phase_offset or not ped)):
                    if len(line.split('=')) > 1:
                        phase_offset = float(line.split('=')[-1].strip())
                        if (np.abs(phase_offset) > 1 and
                                np.abs(phase_offset) < 3):
                            ped = 'ROW'
                        elif (np.abs(phase_offset) < 1 or
                                np.abs(phase_offset) > 3):
                            ped = 'COL'
                            if np.abs(phase_offset) > 3:
                                phase_offset = -1
                            else:
                                phase_offset = 1
                elif 'lDiffDirections' in line:
                    dwi_directions = float(line.split('=')[-1].strip())
        if multivol:
            if dwi_directions:
                n_vols = dwi_directions
            else:
                n_vols = len(list_dicom)
            real_duration = n_vols * tr

        # Read header from first DICOM file in list
        hd = pydicom.read_file(list_dicom[0])
        try:
            start_time = hd.AcquisitionTime
        except AttributeError:
            try:
                start_time = str(hd.AcquisitionDateTime)[8:]
            except AttributeError:
                raise BananaMissingHeaderValue(
                    'No acquisition time found for this scan.')
        # Get echo times
        num_echoes = hd.EchoTrainLength
        if num_echoes == 1:
            echo_times = [hd.EchoTime]
        else:
            echo_times = set()
            for f in list_dicom:
                hdr = pydicom.read_file(f)
                echo_times.add(hdr.EchoTime)
                if len(echo_times) == num_echoes:
                    break
            echo_times = list(echo_times)
        # Get the orientation of the main magnetic field as a vector
        img_orient = np.reshape(np.asarray(hd.ImageOrientationPatient),
                                newshape=(2, 3))
        b0_orient = np.cross(img_orient[0], img_orient[1])
        # Get voxel sizes
        vox_sizes = list(hd.PixelSpacing)
        vox_sizes.append(hd.SliceThickness)
        # Save extracted values to output dictionary
        self.dict_output['start_time'] = float(start_time)
        self.dict_output['tr'] = float(tr) / 1000.0  # Convert to seconds
        self.dict_output['echo_times'] = [float(t) / 1000.0  # Convert to secs
                                          for t in echo_times]
        self.dict_output['voxel_sizes'] = vox_sizes
        self.dict_output['H'] = list(b0_orient)
        self.dict_output['B0'] = hd.MagneticFieldStrength
        self.dict_output['total_duration'] = float(total_duration)
        self.dict_output['real_duration'] = float(real_duration)
        self.dict_output['ped'] = ped
        self.dict_output['pe_angle'] = str(phase_offset)
        keys = ['start_time', 'tr', 'total_duration', 'real_duration', 'ped',
                'pe_angle']
        with open('scan_header_info.txt', 'w') as f:
                f.write(str(out_name) + '\n')
                for k in keys:
                    f.write(k + ' ' + str(self.dict_output[k]) + '\n')
                f.close()
        if self.inputs.reference:
            os.mkdir('reference_motion_mats')
            np.savetxt('reference_motion_mats/reference_motion_mat.mat',
                       np.eye(4))
            np.savetxt('reference_motion_mats/reference_motion_mat_inv.mat',
                       np.eye(4))

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs.update(self.dict_output)
#         outputs["start_time"] = self.dict_output['start_time']
#         outputs["tr"] = self.dict_output['tr']
#         outputs["total_duration"] = self.dict_output['total_duration']
#         outputs["real_duration"] = self.dict_output['real_duration']
#         outputs["ped"] = self.dict_output['ped']
#         outputs["pe_angle"] = self.dict_output['pe_angle']
        outputs["dcm_info"] = os.getcwd() + '/scan_header_info.txt'
        if self.inputs.reference:
            outputs["ref_motion_mats"] = os.getcwd() + '/reference_motion_mats'

        return outputs

    def get_phase_encoding_direction(self, dicom_path):

        dcm = pydicom.read_file(dicom_path)
        inplane_pe_dir = dcm[int('00181312', 16)].value
        csa_str = dcm[int('00291010', 16)].value
        csa_tr = csareader.read(csa_str)
        pedp = csa_tr['tags']['PhaseEncodingDirectionPositive']['items'][0]
        sign = PEDP_TO_SIGN[pedp]
        return sign, inplane_pe_dir


class ScanTimesInfoInputSpec(BaseInterfaceInputSpec):

    dicom_infos = traits.List(desc='List of dicoms to calculate the difference'
                              ' between consecutive scan start times.')


class ScanTimesInfoOutputSpec(TraitedSpec):

    scan_time_infos = File(exists=True, desc='Text file with scan time '
                           'information')


class ScanTimesInfo(BaseInterface):

    input_spec = ScanTimesInfoInputSpec
    output_spec = ScanTimesInfoOutputSpec

    def _run_interface(self, runtime):
        start_times = []
        for dcm in self.inputs.dicom_infos:
            dcm_info = []
            with open(dcm, 'r') as f:
                for line in f:
                    dcm_info.append(line.strip())
                f.close()
            start_times.append((dcm_info[0], dcm_info[1].split()[-1],
                                dcm_info[4].split()[-1]))
        start_times = sorted(start_times, key=lambda k: k[1])
        time_info = {}
        for i in range(1, len(start_times)):
            time_info[start_times[i-1][0]] = {}
            start = dt.datetime.strptime(start_times[i-1][1], '%H%M%S.%f')
            end = dt.datetime.strptime(start_times[i][1], '%H%M%S.%f')
            duration = float((end-start).total_seconds())
            time_info[start_times[i-1][0]]['scan_duration'] = duration
            time_offset = duration - float(start_times[i-1][2])
            if time_offset < 0:
                time_offset = 0
            time_info[start_times[i-1][0]]['time_offset'] = time_offset
        with open('scan_time_info.txt', 'w') as f:
            for k in list(time_info.keys()):
                f.write(k+' '+str(time_info[k]['scan_duration'])+' ' +
                        str(time_info[k]['time_offset'])+'\n')
            f.close()

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()

        outputs["scan_time_infos"] = os.getcwd()+'/scan_time_info.txt'

        return outputs


class PetTimeInfoInputSpec(BaseInterfaceInputSpec):
    pet_data_dir = Directory(exists=True,
                             desc='Directory the the list-mode data.')


class PetTimeInfoOutputSpec(TraitedSpec):

    pet_end_time = traits.Str(desc='PET end time.')
    pet_start_time = traits.Str(desc='PET start time.')
    pet_duration = traits.Int(desc='PET temporal duration in seconds.')


class PetTimeInfo(BaseInterface):

    input_spec = PetTimeInfoInputSpec
    output_spec = PetTimeInfoOutputSpec

    def _run_interface(self, runtime):
        pet_data_dir = self.inputs.pet_data_dir
        self.dict_output = {}
        pet_duration = None
        for root, dirs, files in os.walk(pet_data_dir):
            bf_files = [f for f in files if not f[0] == '.' and '.bf' in f]
            dirs[:] = [d for d in dirs if not d[0] == '.']

        if not bf_files:
            pet_start_time = None
            pet_endtime = None
            logger.warning(
                'No .bf file found in {}. If you want to perform motion '
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
            with open(pet_image, 'rb') as f:
                for line in f:
                    try:
                        line = line[:-1].decode('utf-8')
                    except UnicodeDecodeError:
                        continue
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
        self.dict_output['pet_endtime'] = pet_endtime
        self.dict_output['pet_duration'] = pet_duration
        self.dict_output['pet_start_time'] = pet_start_time

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()

        outputs["pet_end_time"] = self.dict_output['pet_endtime']
        outputs["pet_start_time"] = self.dict_output['pet_start_time']
        outputs["pet_duration"] = self.dict_output['pet_duration']

        return outputs


class Nii2DicomInputSpec(TraitedSpec):
    in_file = File(mandatory=True, desc='input nifti file')
    reference_dicom = traits.List(mandatory=True, desc='original umap')
#     out_file = Directory(genfile=True, desc='the output dicom file')


class Nii2DicomOutputSpec(TraitedSpec):
    out_file = Directory(exists=True, desc='the output dicom file')


class Nii2Dicom(BaseInterface):
    """
    Creates two umaps in dicom format

    fully compatible with the UTE study:

    Attenuation Correction pipeline

    """

    input_spec = Nii2DicomInputSpec
    output_spec = Nii2DicomOutputSpec

    def _run_interface(self, runtime):
        dcms = self.inputs.reference_dicom
        to_remove = [x for x in dcms if '.dcm' not in x]
        if to_remove:
            for f in to_remove:
                dcms.remove(f)
        nifti_image = nib.load(self.inputs.in_file)
        nii_data = nifti_image.get_data()
        if len(dcms) != nii_data.shape[2]:
            raise Exception('Different number of nifti and dicom files '
                            'provided. Dicom to nifti conversion require the '
                            'same number of files in order to run. Please '
                            'check.')
        os.mkdir('nifti2dicom')
        _, basename, _ = split_filename(self.inputs.in_file)
        for i in range(nii_data.shape[2]):
            dcm = pydicom.read_file(dcms[i])
            nifti = nii_data[:, :, i]
            nifti = nifti.astype('uint16')
            dcm.pixel_array.setflags(write=True)
            dcm.pixel_array.flat[:] = nifti.flat[:]
            dcm.PixelData = dcm.pixel_array.T.tostring()
            dcm.save_as('nifti2dicom/{0}_vol{1}.dcm'
                        .format(basename, str(i).zfill(4)))

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = (
            os.getcwd()+'/nifti2dicom')
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
