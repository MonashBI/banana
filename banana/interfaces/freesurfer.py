from nipype.interfaces.base import (
    BaseInterface, BaseInterfaceInputSpec, traits, File, TraitedSpec,
    Directory, isdefined, CommandLineInputSpec, CommandLine)


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


class AparcStatsInputSpec(CommandLineInputSpec):

    recon_all_dir = Directory(
        argstr='--subjects %',
        desc="Directories containing the recon-all outputs to summarise",
        mandatory=True)
    tablefile = File(argstr='--tablefile %', genfile=True,
                           desc=("(REQUIRED) output table file"))
    hemisphere = traits.Enum(
        'lh', 'rh',
        argstr='--hemi %',
        mandatory=True,
        desc=("(REQUIRED) lh or rh"))
    qdec = traits.Str('fsid', argstr='--qdec %', usedefault=True,
                      desc=("name of the qdec table which has the column of "
                            "subjects ids (fsid)"))
    qdec_long = traits.Str(
        argstr='--qdec-long %',
        desc=("name of the longitudinal qdec table which has the column of tp "
              "ids (fsid) and subject templates (fsid-base)"))
    parc = traits.Enum(
        'aparc', 'aparc.a2009s',
        argstr='--parc %', usedefault=True,
        desc=("parcellation.. default is aparc (alt aparc.a2009s)"))
    measure = traits.Enum(
        'volume', 'thickness', 'thicknessstd', 'meancurv', 'gauscurv',
        'foldind', 'curvind',
        argstr='--measure %',
        desc=("measure: default is area ()"))
    delimiter = traits.Enum(
        'tab', 'comma', 'space', 'semicolon',
        argstr='--delimiter %',
        desc=("delimiter between measures in the table. "
              "default is tab (alt comma, space, semicolon, tab)"))
    skip = traits.Any(argstr='--skip %',
                      desc=("if a subject does not have input, skip it"))
    parcid_only = traits.Bool(
        argstr='--parcid-only',
        desc=("do not pre/append hemi/meas to parcellation name"))
    common_parcs = traits.Bool(
        argstr='--common-parcs',
        desc=("output only the common parcellations of all the subjects "
              "given"))
    parcs_from_file = traits.Str(
        argstr='--parcs-from-file %',
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


class AparcStats(CommandLine):

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
        if isdefined(self.inputs.z):
            fname = self.inputs.z
        else:
            fname = 'aparc_stats.txt'
        return fname
