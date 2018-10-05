import os.path as op
from nipype.interfaces.matlab import MatlabCommand, MatlabInputSpec
from nipype.interfaces.base import TraitedSpec, traits, File, isdefined

from nianalysis.interfaces import MATLAB_RESOURCES


class BaseMaskInputSpec(MatlabInputSpec):

    in_file = File(mandatory=True, desc="Input mask to dialate")
    out_file = File(genfile=True, desc="Name of the output file")
    # Need to override value in input spec to make it non-mandatory
    script = traits.Str(
        argstr='-r \"%s;exit\"',
        desc='m-code to run',
        position=-1)


class BaseMaskOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="Output file")
    raw_output = traits.Str("Raw output of the matlab command")


class BaseMask(MatlabCommand):
    """
    Base class for MATLAB mask interfaces
    """

    input_spec = BaseMaskInputSpec
    output_spec = BaseMaskOutputSpec

    def run(self, **inputs):
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

    def script(self, **inputs):  # @UnusedVariable
        """
        Generate script to perform masking
        """
        raise NotImplementedError

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = self._genfilename('out_file')
        return outputs

    def _gen_filename(self, name):
        if name == 'out_file':
            if isdefined(self.inputs.out_file):
                fname = self.inputs.out_file
            else:
                fname = 'out_file'
        else:
            assert False
        return op.abspath(fname)


class DialateMaskInputSpec(BaseMaskInputSpec):

    dialation = traits.List((traits.Float(), traits.Float(), traits.Float),
                            desc="Size of the dialation")


class DialateMask(BaseMask):
    """
    Base interface for STI classes
    """
    input_spec = DialateMaskInputSpec

    def script(self, **inputs):  # @UnusedVariable
        """
        Generate script to load images, pass them to the STI function along
        with the keyword parameters
        """
        script = """
            % Spherical structure element
            SE = fspecial3('ellipsoid', {dialation});

            % Load whole brain mask
            mask = load_untouch_nii('{in_file}');
            mask = mask.img > 0;
            mask = imdilate(mask, SE > 0);
            save_untouch_nii(mask, '{out_file}');
        """.format(
            dialation=self.inputs.dialation,
            in_file=self.inputs.in_file,
            out_file=self._gen_filename('out_file'))
        return script


class CoilMaskInputSpec(BaseMaskInputSpec):

    dialation = traits.List((traits.Float(), traits.Float(), traits.Float),
                            desc="Size of the dialation")


class CoilMask(BaseMask):
    """
    Generate coil specific masks by thresholding magnitude image
    """

    input_spec = CoilMaskInputSpec

    def script(self, **inputs):  # @UnusedVariable
        """
        Generate script to load images, pass them to the STI function along
        with the keyword parameters
        """
        script = """
            mag = load_untouch_nii('{in_file}');
            whole_brain_mask = load_untouch_nii('{mask_file}');

            % Blur to remove tissue based contrast
            vol = convn(mag.img,SE,'same');
            % Threshold to high-signal area
            vol = vol>graythresh(mag.img);
            % Remove orphaned pixels and then close holes in whole_brain_mask
            vol = imclose(vol,SE>0)>0;
            vol = imopen(vol,SE>0)>0;

            % Clip to brain whole_brain_mask region
            mask = mag;
            mask.img = (vol .* whole_brain_mask) > 0;

            save_untouch_nii(mask, '{out_file}');
        """.format(
            in_file=self.inputs.in_file,
            mask_file=self.inputs.mask,
            out_file=self._genfilename('out_file'))
        return script


class MedianInMasksInputSpec(MatlabInputSpec):

    channels = traits.List(File(), mandatory=True,
                           desc="Input mask to dialate")
    channel_masks = traits.List(File(), mandatory=True,
                                desc="Separate masks for each input file")
    whole_brain_mask = File(mandatory=True, desc="Whole brain mask")
    out_file = File(genfile=True, desc="Name of the output file")
    # Need to override value in input spec to make it non-mandatory
    script = traits.Str(
        argstr='-r \"%s;exit\"',
        desc='m-code to run',
        position=-1)


class MedianInMasks(BaseMask):
    """
    Calculate the median value between voxels across the provided images
    that are not masked out.
    """
    input_spec = MedianInMasksInputSpec

    def script(self, **inputs):  # @UnusedVariable
        """
        Generate script to load images, pass them to the STI function along
        with the keyword parameters
        """
        script = """
            brainMask = load_untouch_nii('{mask}');
            brainMask = brainMask.img>0;

            qsmVol = [];
            missingValues = [];
            dims = [];

            channel_files = {{{channel_files}};
            mask_files = {{{mask_files}}};

            for i=1:length(channel_files)
                nii = load_untouch_nii(channel_files(i));
                mask = load_untouch_nii(mask_files(i));

                qsmVol(:,i) = nii.img(:).*(mask.img(:)>0)-99*(mask.img(:)==0);
                missingValues(:,i) = mask.img(:)==0;

                if isempty(dims)
                    dims = size(nii.img);
                end
            end

            % Order values so median value is at index 16
            qsmVol = sort(qsmVol,2);

            % Adjust median index (16) based on missing values
            indVol=sub2ind(size(qsmVol),1:size(qsmVol,1),nCoils*nEchos-floor(0.5*(nCoils*nEchos-sum(missingValues,2)')));

            % Take median value using index, resize and mask out background
            medVol=reshape(qsmVol(indVol),dims);
            medVol(medVol==-99) = 0;
            medVol(brainMask==0) = 0;

            % Save output
            nii.img = medVol;
            save_untouch_nii(nii,'{out_file}');
        """.format(
            channel_files=','.join(
                "'{}'".format(f) for f in self.inputs.channels),
            mask_files=','.join(
                "'{}'".format(f) for f in self.inputs.channel_masks),
            mask=self.inputs.whole_brain_mask,
            out_file=self._genfilename('out_file'))
        return script
