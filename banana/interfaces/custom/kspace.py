import os.path as op
from nipype.interfaces.matlab import MatlabCommand, MatlabInputSpec
from nipype.interfaces.base import TraitedSpec, traits, File
from banana.interfaces import MATLAB_RESOURCES


class BaseKspaceInputSpec(MatlabInputSpec):

    # Need to override value in input spec to make it non-mandatory
    script = traits.Str(
        argstr='-r \"%s;exit\"',
        desc='m-code to run',
        position=-1)


class BaseKspaceOutputSpec(TraitedSpec):

    out_file = File(exists=True, desc="Processed file")
    raw_output = traits.Str("Raw output of the matlab command")


class BaseKspace(MatlabCommand):
    """
    Base class for MATLAB mask interfaces
    """
    output_spec = BaseKspaceOutputSpec

    def run(self, **inputs):
        self.work_dir = inputs['cwd']
        # Set the script input of the matlab spec
        self.inputs.script = (
            "set_param(0,'CharacterEncoding','UTF-8');\n"
            "addpath(genpath('{}'));\n".format(MATLAB_RESOURCES) +
            self.script(**inputs) +
            "exit;")
        results = super().run(**inputs)
        stdout = results.runtime.stdout
        # Attach stdout to outputs to access matlab results
        results.outputs.raw_output = stdout
        return results

    def script(self, **inputs):
        """
        Generate script to perform masking
        """
        raise NotImplementedError

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = self.out_file
        return outputs

    @property
    def out_file(self):
        return op.realpath(op.abspath(op.join(self.work_dir, 'out_file.nii')))


class LoadSiemensKspaceInputSpec(BaseKspaceInputSpec):
    in_file = File(exists=True, mandatory=True)


class LoadSiemensKspace(BaseKspace):
    """
    Reads a Siemens multi-channel k-space file and saves it in a Matlab
    file with the following variables:

    Output Matlab File Structure
    ----------------------------
    calib_scan : 5-d matrix
        Data from calibration scan organised in the following dimension order:
        channel, freq-encode, phase-encode, partition-encode (slice), echoes
    data_scan : 5-d matrix
        Data from "data" scan organised in dimensions order as calibration scan
    num_phase : int
        The number of phase encodings in the "full" reconstruction. NB: this
        can differ from the length of the 3rd dimension in the data matrix due
        to partial Fourier encoding strategies
    num_partitions : int
        The number of partitions (slices) in the "full" reconstruction. See
         'num_phase' re partial Fourier encoding
    """
    input_spec = LoadSiemensKspaceInputSpec

    def script(self, **inputs):
        """
        Generate script to load Siemens format k-space and save as Matlab
        arrays
        """
        script = """
            data_obj = mapVBVD({in_file},'removeOS');
            % Pick largest data object in file
            if length(data_obj)>1
                multi_obj = data_obj;
                acq_length = cellfun(@(x) x.image.NAcq, multi_obj);
                [~,ind] = max(acq_length);
                data_obj = data_obj{{ind}};
            end

            calib_scan = permute(data_obj.refscan{{''}}, [2, 1, 3, 4, 5]);
            data_scan = permute(data_obj.image{{''}}, [2, 1, 3, 4, 5]);
            num_phase = data_obj.hdr.Config.NPeFTLen;
            num_partitions = data_obj.hdr.Config.NImagePar;
            save({out_file}, calib_scan, data_scan, num_phase, num_partitions)
            """.format(in_file=self.inputs.in_file,
                       out_file=self.inputs.calib_file)
        return script
