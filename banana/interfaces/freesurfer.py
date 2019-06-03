import os.path as op
from nipype.interfaces.base import (
    BaseInterface, BaseInterfaceInputSpec, traits, File, TraitedSpec,
    Directory, isdefined)
from nipype.interfaces.freesurfer.base import FSCommand, FSTraitedSpec


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


class AparcStatsInputSpec(FSTraitedSpec):

    subjects = traits.List(
        traits.Str,
        argstr='--subjects %s',
        desc="Directories containing the recon-all outputs to summarise")

    tablefile = File(argstr='--tablefile %s', genfile=True,
                     desc=("(REQUIRED) output table file"))
    hemisphere = traits.Enum(
        'lh', 'rh',
        argstr='--hemi %s',
        mandatory=True,
        desc=("(REQUIRED) lh or rh"))
    qdec = traits.Str('fsid', argstr='--qdec %s',
                      desc=("name of the qdec table which has the column of "
                            "subjects ids (fsid)"))
    qdec_long = traits.Str(
        argstr='--qdec-long %s',
        desc=("name of the longitudinal qdec table which has the column of tp "
              "ids (fsid) and subject templates (fsid-base)"))
    parc = traits.Enum(
        'aparc', 'aparc.a2009s', 'aparc.DKTatlas40',
        argstr='--parc %s', usedefault=True,
        desc=("parcellation.. default is aparc (alt aparc.a2009s)"))
    measure = traits.Enum(
        'volume', 'thickness', 'thicknessstd', 'meancurv', 'gauscurv',
        'foldind', 'curvind',
        argstr='--measure %s',
        desc=("measure: default is area ()"))
    delimiter = traits.Enum(
        'tab', 'comma', 'space', 'semicolon',
        argstr='--delimiter %s',
        desc=("delimiter between measures in the table. "
              "default is tab (alt comma, space, semicolon, tab)"))
    skip = traits.Any(argstr='--skip %s',
                      desc=("if a subject does not have input, skip it"))
    parcid_only = traits.Bool(
        argstr='--parcid-only',
        desc=("do not pre/append hemi/meas to parcellation name"))
    common_parcs = traits.Bool(
        argstr='--common-parcs',
        desc=("output only the common parcellations of all the subjects "
              "given"))
    parcs_from_file = traits.Str(
        argstr='--parcs-from-file %s',
        desc=("filename: output parcellations specified in the file"))
    report_rois = traits.Bool(
        argstr='--report-rois',
        desc=("print ROIs information for each subject"))
    transpose = traits.Bool(
        argstr='--transpose',
        desc=("transpose the table ( default is subjects in rows and ROIs in "
              "cols)"))


class AparcStatsOutputSpec(TraitedSpec):

    tablefile = File(exists=True, desc="The output table file")


class AparcStats(FSCommand):

    input_spec = AparcStatsInputSpec
    output_spec = AparcStatsOutputSpec
    _cmd = 'aparcstats2table'

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['tablefile'] = self._gen_tablefile_fname()
        return outputs

    def _gen_filename(self, name):
        if name == 'tablefile':
            fname = self._gen_tablefile_fname()
        else:
            assert False
        return fname

    def _gen_tablefile_fname(self):
        if isdefined(self.inputs.tablefile):
            fname = self.inputs.tablefile
        else:
            fname = 'aparc_{}_{}_table.txt'.format(self.inputs.hemisphere,
                                                   self.inputs.measure)
        return op.abspath(fname)
