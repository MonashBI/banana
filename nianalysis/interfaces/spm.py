# Standard library imports
import os

# Third-party imports
import numpy as np

# Local imports
from nipype.interfaces.base import (OutputMultiPath, TraitedSpec, isdefined,
                                    traits, InputMultiPath, File)
from nipype.interfaces.spm.base import (SPMCommand, scans_for_fnames,
                                        SPMCommandInputSpec)
from nipype.utils.filemanip import split_filename


class MultiChannelSegmentInputSpec(SPMCommandInputSpec):
    channel_files = InputMultiPath(File(exists=True),
                                   desc="A list of files to be segmented",
                                   field='channel', copyfile=False,
                                   mandatory=True)
    channel_info = traits.Tuple(traits.Float(), traits.Float(),
                                traits.Tuple(traits.Bool, traits.Bool),
                                desc="""A tuple with the following fields:
            - bias reguralisation (0-10)
            - FWHM of Gaussian smoothness of bias
            - which maps to save (Corrected, Field) - a tuple of two boolean values""",  # @IgnorePep8
                                field='channel')
    tissues = traits.List(
        traits.Tuple(traits.Tuple(File(exists=True), traits.Int()),
                     traits.Int(), traits.Tuple(traits.Bool, traits.Bool),
                     traits.Tuple(traits.Bool, traits.Bool)),
        desc="""A list of tuples (one per tissue) with the following fields:
            - tissue probability map (4D), 1-based index to frame
            - number of gaussians
            - which maps to save [Native, DARTEL] - a tuple of two boolean values
            - which maps to save [Unmodulated, Modulated] - a tuple of two boolean values""",  # @IgnorePep8
        field='tissue')
    affine_regularization = traits.Enum('mni', 'eastern', 'subj', 'none',
                                        field='warp.affreg',
                                        desc='mni, eastern, subj, none ')
    warping_regularization = traits.Either(traits.List(traits.Float(),
                                                       minlen=5, maxlen=5),
                                           traits.Float(),
                                           field='warp.reg',
                                           desc=('Warping regularization '
                                                 'parameter(s). Accepts float '
                                                 'or list of floats (the '
                                                 'latter is required by '
                                                 'SPM12)'))
    sampling_distance = traits.Float(field='warp.samp',
                                     desc=('Sampling distance on data for '
                                           'parameter estimation'))
    write_deformation_fields = traits.List(traits.Bool(), minlen=2, maxlen=2,
                                           field='warp.write',
                                           desc=("Which deformation fields to "
                                                 "write:[Inverse, Forward]"))


class MultiChannelSegmentOutputSpec(TraitedSpec):
    native_class_images = traits.List(traits.List(File(exists=True)),
                                      desc='native space probability maps')
    dartel_input_images = traits.List(traits.List(File(exists=True)),
                                      desc='dartel imported class images')
    normalized_class_images = traits.List(traits.List(File(exists=True)),
                                          desc='normalized class images')
    modulated_class_images = traits.List(traits.List(File(exists=True)),
                                         desc=('modulated+normalized class '
                                               'images'))
    transformation_mat = OutputMultiPath(File(exists=True),
                                         desc='Normalization transformation')
    bias_corrected_images = OutputMultiPath(File(exists=True),
                                            desc='bias corrected images')
    bias_field_images = OutputMultiPath(File(exists=True),
                                        desc='bias field images')
    forward_deformation_field = OutputMultiPath(File(exists=True))
    inverse_deformation_field = OutputMultiPath(File(exists=True))


class MultiChannelSegment(SPMCommand):
    """Use spm_preproc8 (New Segment) to separate structural images into
    different tissue classes. Supports multiple modalities.

    NOTE: This interface currently supports single channel input only

    http://www.fil.ion.ucl.ac.uk/spm/doc/manual.pdf#page=43

    Examples
    --------
    >>> import nipype.interfaces.spm as spm
    >>> seg = spm.Segment()
    >>> seg.inputs.channel_files = 'structural.nii'
    >>> seg.inputs.channel_info = (0.0001, 60, (True, True))
    >>> seg.run() # doctest: +SKIP

    For VBM pre-processing [http://www.fil.ion.ucl.ac.uk/~john/misc/VBMclass10.pdf],
    TPM.nii should be replaced by /path/to/spm8/toolbox/Seg/TPM.nii

    >>> seg = Segment()
    >>> seg.inputs.channel_files = 'structural.nii'
    >>> tissue1 = (('TPM.nii', 1), 2, (True,True), (False, False))
    >>> tissue2 = (('TPM.nii', 2), 2, (True,True), (False, False))
    >>> tissue3 = (('TPM.nii', 3), 2, (True,False), (False, False))
    >>> tissue4 = (('TPM.nii', 4), 2, (False,False), (False, False))
    >>> tissue5 = (('TPM.nii', 5), 2, (False,False), (False, False))
    >>> seg.inputs.tissues = [tissue1, tissue2, tissue3, tissue4, tissue5]
    >>> seg.run() # doctest: +SKIP

    """

    input_spec = MultiChannelSegmentInputSpec
    output_spec = MultiChannelSegmentOutputSpec

    def __init__(self, **inputs):
        _local_version = SPMCommand().version
        if _local_version and '12.' in _local_version:
            self._jobtype = 'spatial'
            self._jobname = 'preproc'
        else:
            self._jobtype = 'tools'
            self._jobname = 'preproc8'

        SPMCommand.__init__(self, **inputs)

    def _format_arg(self, opt, spec, val):
        """Convert input to appropriate format for spm
        """

        if opt in ['channel_files', 'channel_info']:
            # structure have to be recreated because of some weird traits error
            new_channel = {}
            new_channel['vols'] = scans_for_fnames(self.inputs.channel_files)
            if isdefined(self.inputs.channel_info):
                info = self.inputs.channel_info
                new_channel['biasreg'] = info[0]
                new_channel['biasfwhm'] = info[1]
                new_channel['write'] = [int(info[2][0]), int(info[2][1])]
            return [new_channel]
        elif opt == 'tissues':
            new_tissues = []
            for tissue in val:
                new_tissue = {}
                new_tissue['tpm'] = np.array([','.join([tissue[0][0],
                                                        str(tissue[0][1])])],
                                             dtype=object)
                new_tissue['ngaus'] = tissue[1]
                new_tissue['native'] = [int(tissue[2][0]), int(tissue[2][1])]
                new_tissue['warped'] = [int(tissue[3][0]), int(tissue[3][1])]
                new_tissues.append(new_tissue)
            return new_tissues
        elif opt == 'write_deformation_fields':
            return super(MultiChannelSegment, self)._format_arg(opt, spec,
                                                                [int(val[0]),
                                                                 int(val[1])])
        else:
            return super(MultiChannelSegment, self)._format_arg(opt, spec, val)

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['native_class_images'] = []
        outputs['dartel_input_images'] = []
        outputs['normalized_class_images'] = []
        outputs['modulated_class_images'] = []
        outputs['transformation_mat'] = []
        outputs['bias_corrected_images'] = []
        outputs['bias_field_images'] = []
        outputs['inverse_deformation_field'] = []
        outputs['forward_deformation_field'] = []

        n_classes = 5
        if isdefined(self.inputs.tissues):
            n_classes = len(self.inputs.tissues)
        for i in range(n_classes):
            outputs['native_class_images'].append([])
            outputs['dartel_input_images'].append([])
            outputs['normalized_class_images'].append([])
            outputs['modulated_class_images'].append([])

        for filename in self.inputs.channel_files:
            pth, base, ext = split_filename(filename)
            if isdefined(self.inputs.tissues):
                for i, tissue in enumerate(self.inputs.tissues):
                    if tissue[2][0]:
                        outputs['native_class_images'][i].append(
                            os.path.join(pth, "c%d%s.nii" % (i + 1, base)))
                    if tissue[2][1]:
                        outputs['dartel_input_images'][i].append(
                            os.path.join(pth, "rc%d%s.nii" % (i + 1, base)))
                    if tissue[3][0]:
                        outputs['normalized_class_images'][i].append(
                            os.path.join(pth, "wc%d%s.nii" % (i + 1, base)))
                    if tissue[3][1]:
                        outputs['modulated_class_images'][i].append(
                            os.path.join(pth, "mwc%d%s.nii" % (i + 1, base)))
            else:
                for i in range(n_classes):
                    outputs['native_class_images'][i].append(
                        os.path.join(pth, "c%d%s.nii" % (i + 1, base)))
            outputs['transformation_mat'].append(
                os.path.join(pth, "%s_seg8.mat" % base))

            if isdefined(self.inputs.write_deformation_fields):
                if self.inputs.write_deformation_fields[0]:
                    outputs['inverse_deformation_field'].append(
                        os.path.join(pth, "iy_%s.nii" % base))
                if self.inputs.write_deformation_fields[1]:
                    outputs['forward_deformation_field'].append(
                        os.path.join(pth, "y_%s.nii" % base))

            if isdefined(self.inputs.channel_info):
                if self.inputs.channel_info[2][0]:
                    outputs['bias_corrected_images'].append(
                        os.path.join(pth, "m%s.nii" % (base)))
                if self.inputs.channel_info[2][1]:
                    outputs['bias_field_images'].append(
                        os.path.join(pth, "BiasField_%s.nii" % (base)))
        return outputs
