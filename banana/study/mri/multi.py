"""
Common multi-study MultiStudy combinations
"""
from banana import MultiStudy, MultiStudyMetaClass, SubStudySpec
from .dwi import DwiStudy
from .t1w import T1wStudy
from .t2w import T2wStudy


class DwiAndT1wStudy(MultiStudy, metaclass=MultiStudyMetaClass):

    desc = ("A DWI series with a T1-weighted contrast images co-registered to "
            "it provide anatomical constraints to the tractography and "
            "generate a connectome")

    add_substudy_specs = [
        SubStudySpec(
            'dwi',
            DwiStudy,
            name_map={
                'anat_5tt': 't1_five_tissue_type',
                'anat_fs_recon_all': 't1_fs_recon_all'}),
        SubStudySpec(
            't1',
            T1wStudy,
            name_map={
                'coreg_ref': 'dwi_mag_preproc'})]


class T1AndT2wStudy(MultiStudy, metaclass=MultiStudyMetaClass):

    desc = ("A T1-weighted contrast with a T2-weighted contrast co-registered "
            "to it to improve the segmentation of the peel surface in "
            "Freesurfer")

    add_substudy_specs = [
        SubStudySpec(
            't1',
            T1wStudy,
            name_map={
                't2_coreg': 't2_mag_coreg'}),
        SubStudySpec(
            't2',
            T2wStudy,
            name_map={
                'coreg_ref': 't1_magnitude'})]
