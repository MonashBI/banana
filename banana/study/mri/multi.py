"""
Common multi-study MultiAnalysis combinations
"""
from banana import MultiAnalysis, MultiAnalysisMetaClass, SubAnalysisSpec
from .dwi import DwiAnalysis
from .t1w import T1wAnalysis
from .t2w import T2wAnalysis


class DwiAndT1wAnalysis(MultiAnalysis, metaclass=MultiAnalysisMetaClass):

    desc = ("A DWI series with a T1-weighted contrast images co-registered to "
            "it provide anatomical constraints to the tractography and "
            "generate a connectome")

    add_substudy_specs = [
        SubAnalysisSpec(
            'dwi',
            DwiAnalysis,
            name_map={
                'anat_5tt': 't1_five_tissue_type',
                'anat_fs_recon_all': 't1_fs_recon_all'}),
        SubAnalysisSpec(
            't1',
            T1wAnalysis,
            name_map={
                'coreg_ref': 'dwi_mag_preproc'})]


class T1AndT2wAnalysis(MultiAnalysis, metaclass=MultiAnalysisMetaClass):

    desc = ("A T1-weighted contrast with a T2-weighted contrast co-registered "
            "to it to improve the segmentation of the peel surface in "
            "Freesurfer")

    add_substudy_specs = [
        SubAnalysisSpec(
            't1',
            T1wAnalysis,
            name_map={
                't2_coreg': 't2_mag_coreg'}),
        SubAnalysisSpec(
            't2',
            T2wAnalysis,
            name_map={
                'coreg_ref': 't1_magnitude'})]
