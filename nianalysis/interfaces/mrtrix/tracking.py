from nipype.interfaces.base import traits, File, TraitedSpec
from nipype.interfaces.mrtrix3.reconst import (
    MRTrix3Base, MRTrix3BaseInputSpec)


class GlobalTractographyInputSpec(MRTrix3BaseInputSpec):
    in_file = File(exists=True, argstr='%s', mandatory=True,
                    position=-3, desc='input dMRI file')
    wm_response = File(exists=True, argstr='%s', mandatory=True,
                       position=-2)
    out_file = File(name_template='%s_tracks.tck',
                    name_source='in_file',
                    argstr='%s',
                    desc=("Output streamlines"), position=-1)
    num_iterations = traits.Int(1e9, argstr='-niter %s',
                                usedefault=True,
                                desc="number of tracks to generate")
    mask = File(exists=True, argstr='-mask %s', mandatory=False,
                desc="Brain mask")


class GlobalTractographyOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc=("Output streamlines"))


class GlobalTractography(MRTrix3Base):
    """Global tractography"""

    _cmd = "tckglobal"
    input_spec = GlobalTractographyInputSpec
    output_spec = GlobalTractographyOutputSpec
