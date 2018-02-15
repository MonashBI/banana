
from __future__ import absolute_import
from nipype.interfaces.base import (BaseInterface, BaseInterfaceInputSpec,
                                    traits, TraitedSpec, Directory)
import numpy as np
import glob
import dicom


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
    pe_angle = traits.Str(desc='Phase angle.')


class DicomHeaderInfoExtraction(BaseInterface):

    input_spec = DicomHeaderInfoExtractionInputSpec
    output_spec = DicomHeaderInfoExtractionOutputSpec

    def _run_interface(self, runtime):

        list_dicom = sorted(glob.glob(self.inputs.dicom_folder+'/*'))
        multivol = self.inputs.multivol
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
                        if np.abs(phase_offset) > 1 and np.abs(phase_offset) < 3:
                            ped = 'ROW'
                        elif np.abs(phase_offset) < 1 or np.abs(phase_offset) > 3:
                            ped = 'COL'
                elif 'lDiffDirections' in line:
                    dwi_directions = float(line.split('=')[-1].strip())
        if multivol:
            if dwi_directions:
                n_vols = dwi_directions
            else:
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
        self.dict_output['start_time'] = str(start_time)
        self.dict_output['tr'] = tr
        self.dict_output['total_duration'] = str(total_duration)
        self.dict_output['real_duration'] = str(real_duration)
        self.dict_output['ped'] = ped
        self.dict_output['pe_angle'] = str(phase_offset)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()

        outputs["start_time"] = self.dict_output['start_time']
        outputs["tr"] = self.dict_output['tr']
        outputs["tot_duration"] = self.dict_output['total_duration']
        outputs["real_duration"] = self.dict_output['real_duration']
        outputs["ped"] = self.dict_output['ped']
        outputs["pe_angle"] = self.dict_output['pe_angle']

        return outputs
