
from __future__ import absolute_import
from nipype.interfaces.base import (BaseInterface, BaseInterfaceInputSpec,
                                    traits, TraitedSpec, Directory, File)
import numpy as np
import glob
import pydicom
from nipype.utils.filemanip import split_filename
import os
import datetime as dt


class DicomHeaderInfoExtractionInputSpec(BaseInterfaceInputSpec):

    dicom_folder = Directory(exists=True, desc='Directory with DICOM files',
                             mandatory=True)
    multivol = traits.Bool(desc='Specify whether a scan is 3D or 4D',
                           default=False)
    reference = traits.Bool(desc='Specify whether the input scan is the motion'
                            ' correction reference.')


class DicomHeaderInfoExtractionOutputSpec(TraitedSpec):

    tr = traits.Float(desc='Repetition time.')
    start_time = traits.Str(desc='Scan start time.')
    real_duration = traits.Str(desc='For 4D files, this will be the number of '
                               'volumes multiplied by the TR.')
    tot_duration = traits.Str(
        desc='Scan duration as extracted from the header.')
    ped = traits.Str(desc='Phase encoding direction.')
    pe_angle = traits.Str(desc='Phase angle.')
    dcm_info = File(exists=True, desc='File with all the previous outputs.')
    ref_motion_mats = Directory(desc='folder with the reference motion mats')


class DicomHeaderInfoExtraction(BaseInterface):

    input_spec = DicomHeaderInfoExtractionInputSpec
    output_spec = DicomHeaderInfoExtractionOutputSpec

    def _run_interface(self, runtime):

        list_dicom = sorted(glob.glob(self.inputs.dicom_folder+'/*'))
        multivol = self.inputs.multivol
        _, out_name, _ = split_filename(self.inputs.dicom_folder)
        ped = ''
        phase_offset = ''
        self.dict_output = {}
        dwi_directions = None

        with open(list_dicom[0], 'r') as f:
            for line in f:
                if 'TotalScan' in line:
                    total_duration = line.split('=')[-1].strip()
                    if not multivol:
                        real_duration = total_duration
                elif 'alTR[0]' in line:
                    tr = float(line.split('=')[-1].strip())/1000000
                elif 'SliceArray.asSlice[0].dInPlaneRot' in line and multivol:
                    if len(line.split('=')) > 1:
                        phase_offset = float(line.split('=')[-1].strip())
                        if (np.abs(phase_offset) > 1 and
                                np.abs(phase_offset) < 3):
                            ped = 'ROW'
                        elif (np.abs(phase_offset) < 1 or
                                np.abs(phase_offset) > 3):
                            ped = 'COL'
                elif 'lDiffDirections' in line:
                    dwi_directions = float(line.split('=')[-1].strip())
        if multivol:
            if dwi_directions:
                n_vols = dwi_directions
            else:
                n_vols = len(list_dicom)
            real_duration = n_vols*tr

        hd = pydicom.read_file(list_dicom[0])
        try:
            start_time = str(hd.AcquisitionTime)
        except AttributeError:
            try:
                start_time = str(hd.AcquisitionDateTime)[8:]
            except AttributeError:
                raise Exception('No acquisition time found for this scan.')
        self.dict_output['start_time'] = str(start_time)
        self.dict_output['tr'] = tr
        self.dict_output['total_duration'] = str(total_duration)
        self.dict_output['real_duration'] = str(real_duration)
        self.dict_output['ped'] = ped
        self.dict_output['pe_angle'] = str(phase_offset)
        keys = ['start_time', 'tr', 'total_duration', 'real_duration', 'ped',
                'pe_angle']
        with open('scan_header_info.txt', 'w') as f:
                f.write(str(out_name)+'\n')
                for k in keys:
                    f.write(k+' '+str(self.dict_output[k])+'\n')
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

        outputs["start_time"] = self.dict_output['start_time']
        outputs["tr"] = self.dict_output['tr']
        outputs["tot_duration"] = self.dict_output['total_duration']
        outputs["real_duration"] = self.dict_output['real_duration']
        outputs["ped"] = self.dict_output['ped']
        outputs["pe_angle"] = self.dict_output['pe_angle']
        outputs["dcm_info"] = os.getcwd()+'/scan_header_info.txt'
        if self.inputs.reference:
            outputs["ref_motion_mats"] = os.getcwd()+'/reference_motion_mats'

        return outputs


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
#         start_times = [(x.keys()[0], x[x.keys()[0]]['start_time'],
#                         x[x.keys()[0]]['real_duration']) for x in
#                        self.inputs.dicom_infos]
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
            for k in time_info.keys():
                f.write(k+' '+str(time_info[k]['scan_duration'])+' ' +
                        str(time_info[k]['time_offset'])+'\n')
            f.close()

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()

        outputs["scan_time_infos"] = os.getcwd()+'/scan_time_info.txt'

        return outputs
