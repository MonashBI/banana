from nipype.interfaces.base import (BaseInterface, BaseInterfaceInputSpec,
                                    traits, File, TraitedSpec, Directory)


class GenFSBrainMasksInputSpec(BaseInterfaceInputSpec):

    fs_directory = Directory(exists=True, mandatory=True, desc='Directory with'
                             'freesurfer results.')


class GenFSBrainMasksOutputSpec(TraitedSpec):

    totGM = File(exists=True, desc='Total grey matter mask')
    corticalGM = File(exists=True, desc='Cortical grey matter mask')
    totWM = File(exists=True, desc='Total white matter mask')
    subcorticalGM = File(exists=True, desc='Subcortical grey matter mask')
    totCSF = File(exists=True, desc='Total CSF mask')
    totBrainVol = File(exists=True, desc='Total brain volume mask')
    totBrainVolNoVent = File(exists=True, desc='Total brain volume mask'
                             'without ventricles')
    Brainstem = File(exists=True, desc='Brainstem mask')


class GenFSBrainMasks(BaseInterface):

    input_spec = GenFSBrainMasksInputSpec
    output_spec = GenFSBrainMasksOutputSpec

    def _run_interface(self, runtime):
        pass
