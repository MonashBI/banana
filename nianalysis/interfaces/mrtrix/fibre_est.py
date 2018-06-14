import os.path
from nipype.interfaces.base import (
    traits, InputMultiPath, File, TraitedSpec, isdefined)
from nipype.interfaces.mrtrix3.reconst import (
    MRTrix3Base, MRTrix3BaseInputSpec)
from nipype.interfaces.mrtrix3.preprocess import (
    ResponseSD as NipypeResponseSD,
    ResponseSDInputSpec as NipypeResponseSDInputSpec)
from arcana.utils import split_extension


class Fod2FixelInputSpec(MRTrix3BaseInputSpec):
    in_file = File(exists=True, argstr='%s', mandatory=True, position=-3,
                   desc='input files')
    out_file = File(genfile=True, argstr='%s', desc=(""), position=-1)


class Fod2FixelOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc=(""))


class Fod2Fixel(MRTrix3Base):
    """Fod2Fixel"""

    _cmd = "fod2fixel"
    input_spec = Fod2FixelInputSpec
    output_spec = Fod2FixelOutputSpec

    def _list_outputs(self):

        outputs = self.output_spec().get()
        return outputs


class Fixel2VoxelInputSpec(MRTrix3BaseInputSpec):
    in_file = File(exists=True, argstr='%s', mandatory=True, position=-3,
                   desc='input files')
    out_file = File(genfile=True, argstr='%s', desc=(""), position=-1)


class Fixel2VoxelOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc=(""))


class Fixel2Voxel(MRTrix3Base):
    """Fixel2Voxel"""

    _cmd = "fixel2voxel"
    input_spec = Fixel2VoxelInputSpec
    output_spec = Fixel2VoxelOutputSpec

    def _list_outputs(self):

        outputs = self.output_spec().get()
        return outputs


class FixelCorrespondenceInputSpec(MRTrix3BaseInputSpec):
    in_file = File(exists=True, argstr='%s', mandatory=True, position=-3,
                   desc='input files')
    out_file = File(genfile=True, argstr='%s', desc=(""), position=-1)


class FixelCorrespondenceOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc=(""))


class FixelCorrespondence(MRTrix3Base):
    """FixelCorrespondence"""

    _cmd = "fixelcorrespondence"
    input_spec = FixelCorrespondenceInputSpec
    output_spec = FixelCorrespondenceOutputSpec

    def _list_outputs(self):

        outputs = self.output_spec().get()
        return outputs


class FixelCFEStatsInputSpec(MRTrix3BaseInputSpec):
    in_file = File(exists=True, argstr='%s', mandatory=True, position=-3,
                   desc='input files')
    out_file = File(genfile=True, argstr='%s', desc=(""), position=-1)


class FixelCFEStatsOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc=(""))


class FixelCFEStats(MRTrix3Base):
    """FixelCFEStats"""

    _cmd = "fixelcfestats"
    input_spec = FixelCFEStatsInputSpec
    output_spec = FixelCFEStatsOutputSpec

    def _list_outputs(self):

        outputs = self.output_spec().get()
        return outputs


class TckSiftInputSpec(MRTrix3BaseInputSpec):
    in_file = File(exists=True, argstr='%s', mandatory=True, position=-3,
                   desc='input files')
    out_file = File(genfile=True, argstr='%s', desc=(""), position=-1)


class TckSiftOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc=(""))


class TckSift(MRTrix3Base):
    """TckSift"""

    _cmd = "tcksift"
    input_spec = TckSiftInputSpec
    output_spec = TckSiftOutputSpec

    def _list_outputs(self):

        outputs = self.output_spec().get()
        return outputs


class Warp2MetricInputSpec(MRTrix3BaseInputSpec):
    in_file = File(exists=True, argstr='%s', mandatory=True, position=-3,
                   desc='input files')
    out_file = File(genfile=True, argstr='%s', desc=(""), position=-1)


class Warp2MetricOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc=(""))


class Warp2Metric(MRTrix3Base):
    """Warp2Metric"""

    _cmd = "warp2metric"
    input_spec = Warp2MetricInputSpec
    output_spec = Warp2MetricOutputSpec

    def _list_outputs(self):

        outputs = self.output_spec().get()
        return outputs


class AverageReponseInputSpec(MRTrix3BaseInputSpec):

    in_files = InputMultiPath(
        File(exists=True), argstr='%s', mandatory=True,
        position=0, desc="Average response")

    out_file = File(
        genfile=True, argstr='%s', position=-1,
        desc=("the output spherical harmonics coefficients image"))


class AverageReponseOutputSpec(TraitedSpec):

    out_file = File(exists=True, desc='the output response file')


class AverageResponse(MRTrix3Base):

    _cmd = 'average_response'
    input_spec = AverageReponseInputSpec
    output_spec = AverageReponseOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = os.path.abspath(self._gen_outfilename())
        return outputs

    def _gen_filename(self, name):
        if name == 'out_file':
            fname = self._gen_outfilename()
        else:
            assert False
        return fname

    def _gen_outfilename(self):
        if isdefined(self.inputs.out_file):
            filename = self.inputs.out_file
        else:
            base, ext = split_extension(
                os.path.basename(self.inputs.in_files[0]))
            filename = os.path.join(os.getcwd(),
                                    "{}_avg{}".format(base, ext))
        return filename


class EstimateFODInputSpec(MRTrix3BaseInputSpec):

    algorithm = traits.Enum('csd', 'msmt_csd', argstr='%s', mandatory=True,
                            position=0,
                            desc="Algorithm used for CSD estimation")

    in_file = File(exists=True, argstr='%s', mandatory=True, position=-3,
                   desc='input diffusion weighted images')
    response = File(
        exists=True, argstr='%s', mandatory=True, position=-2,
        desc=('a text file containing the diffusion-weighted signal response '
              'function coefficients for a single fibre population'))
    out_file = File(
        'fods.mif', argstr='%s', mandatory=True, position=-1,
        usedefault=True, desc=('the output spherical harmonics coefficients'
                               ' image'))

    # DW Shell selection parameters
    shell = traits.List(traits.Float, sep=',', argstr='-shell %s',
                        desc='specify one or more dw gradient shells')

    # Spherical deconvolution parameters
    max_sh = traits.Int(8, argstr='-lmax %d',
                        desc='maximum harmonic degree of response function')
    in_mask = File(exists=True, argstr='-mask %s',
                   desc='provide initial mask image')
    in_dirs = File(
        exists=True, argstr='-directions %s',
        desc=('specify the directions over which to apply the non-negativity '
              'constraint (by default, the built-in 300 direction set is '
              'used). These should be supplied as a text file containing the '
              '[ az el ] pairs for the directions.'))
    sh_filter = File(
        exists=True, argstr='-filter %s',
        desc=('the linear frequency filtering parameters used for the initial '
              'linear spherical deconvolution step (default = [ 1 1 1 0 0 ]). '
              'These should be supplied as a text file containing the '
              'filtering coefficients for each even harmonic order.'))

    neg_lambda = traits.Float(
        1.0, argstr='-neg_lambda %f',
        desc=('the regularisation parameter lambda that controls the strength'
              ' of the non-negativity constraint'))
    thres = traits.Float(
        0.0, argstr='-threshold %f',
        desc=('the threshold below which the amplitude of the FOD is assumed '
              'to be zero, expressed as an absolute amplitude'))

    n_iter = traits.Int(
        50, argstr='-niter %d', desc=('the maximum number of iterations '
                                      'to perform for each voxel'))


class EstimateFODOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='the output response file')


class EstimateFOD(MRTrix3Base):

    """
    Convert diffusion-weighted images to tensor images

    Note that this program makes use of implied symmetries in the diffusion
    profile. First, the fact the signal attenuation profile is real implies
    that it has conjugate symmetry, i.e. Y(l,-m) = Y(l,m)* (where * denotes
    the complex conjugate). Second, the diffusion profile should be
    antipodally symmetric (i.e. S(x) = S(-x)), implying that all odd l
    components should be zero. Therefore, this program only computes the even
    elements.

    Note that the spherical harmonics equations used here differ slightly from
    those conventionally used, in that the (-1)^m factor has been omitted.
    This should be taken into account in all subsequent calculations.
    The spherical harmonic coefficients are stored as follows. First, since
    the signal attenuation profile is real, it has conjugate symmetry, i.e.
    Y(l,-m) = Y(l,m)* (where * denotes the complex conjugate). Second, the
    diffusion profile should be antipodally symmetric (i.e. S(x) = S(-x)),
    implying that all odd l components should be zero. Therefore, only the
    even elements are computed.

    Note that the spherical harmonics equations used here differ slightly from
    those conventionally used, in that the (-1)^m factor has been omitted.
    This should be taken into account in all subsequent calculations.
    Each volume in the output image corresponds to a different spherical
    harmonic component. Each volume will correspond to the following:

    volume 0: l = 0, m = 0
    volume 1: l = 2, m = -2 (imaginary part of m=2 SH)
    volume 2: l = 2, m = -1 (imaginary part of m=1 SH)
    volume 3: l = 2, m = 0
    volume 4: l = 2, m = 1 (real part of m=1 SH)
    volume 5: l = 2, m = 2 (real part of m=2 SH)
    etc...



    Example
    -------

    >>> import nipype.interfaces.mrtrix3 as mrt
    >>> fod = mrt.EstimateFOD()
    >>> fod.inputs.in_file = 'dwi.mif'
    >>> fod.inputs.response = 'response.txt'
    >>> fod.inputs.in_mask = 'mask.nii.gz'
    >>> fod.inputs.grad_fsl = ('bvecs', 'bvals')
    >>> fod.cmdline                               # doctest: +ELLIPSIS +ALLOW_UNICODE
    'dwi2fod -fslgrad bvecs bvals -mask mask.nii.gz dwi.mif response.txt\
 fods.mif'
    >>> fod.run()                                 # doctest: +SKIP
    """

    _cmd = 'dwi2fod'
    input_spec = EstimateFODInputSpec
    output_spec = EstimateFODOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = os.path.abspath(self.inputs.out_file)
        return outputs
