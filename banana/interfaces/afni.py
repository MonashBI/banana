from nipype.interfaces.afni.base import AFNICommand, AFNICommandInputSpec
from nipype.interfaces.base import File, traits, TraitedSpec


class TprojectInputSpec(AFNICommandInputSpec):

    in_file = File(desc='input file to 3dTproject', argstr='-input %s',
                   exists=True, mandatory=True)
    out_file = File(name_template="%s_filt", desc='output image file name',
                    argstr='-prefix %s', name_source="in_file")
    ort = File(desc='remove each column in file. Multiple ort are allowed',
               argstr='-ort %s', exists=True)
    polort = traits.Int(desc='remove polynomials up to degree specified',
                        argstr='-polort %s')
    dsort = File(
        desc='Remove the 3D+time time series in the specified dataset',
        argstr='-dsort %s', exists=True)
    passband = traits.Tuple(
        (traits.Float(), traits.Float()), argstr='-passband %f %f',
        desc='Remove all frequencies EXCEPT those in the range provided.')
    stopband = traits.Tuple(
        (traits.Float(), traits.Float()), argstr='-stopband %f %f',
        desc='Remove all frequencies in the range provided.')
    delta_t = traits.Float(desc='time step for frequency calculations',
                           argstr='-dt %f')
    mask = File(desc='Only operate on voxels nonzero in the provided mask.',
                argstr='-mask %s', exists=True, xor=['automask'])
    automask = traits.Bool(desc='the program will generate the mask to use for'
                           'calculations.', argstr='-automask', xor=['mask'])
    blur = traits.Float(desc='Blur (inside the mask only) with a filter that'
                        'has width (FWHM) of fff millimeters.',
                        argstr='-blur %f')
    normalize = traits.Bool(desc='Normalize each output time series to have'
                            'sum of squares = 1. This is the LAST operation.',
                            argstr='-norm')


class TprojectOutputSpec(TraitedSpec):
    out_file = File(desc='filtered file', exists=True)


class Tproject(AFNICommand):
    """This program projects (detrends) out various 'nuisance' time series from
    each voxel in the input dataset.  Note that all the projections are done
    via linear regression, including the frequency-based parameters such as
    '-passband'.  In this way, you can bandpass time-censored data, and at the
    same time, remove other time series of no interest (e.g., physiological
    estimates, motion parameters).

    Examples
    ========
    >>> banana.interfaces.afni import Tproject
    >>> TP = Tproject()
    >>> TP.inputs.in_file = 'functional.nii'
    >>> TP.inputs.passband = (0.01, 0.08)
    >>> TP.delta_t = 0.754
    >>> TP.inputs.outputtype = "NIFTI"
    >>> res = TP.run()
    """

    _cmd = '3dTproject'
    input_spec = TprojectInputSpec
    output_spec = TprojectOutputSpec
