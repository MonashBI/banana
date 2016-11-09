from argparse import ArgumentParser
import tempfile
import subprocess as sp
import shutil
import re
import os.path
from nianalysis.archive.daris import DarisLogin


parser = ArgumentParser()
parser.add_argument('password', type=str,
                    help="Password for the system/manager account")
parser.add_argument('--subject_ids', nargs='+', default=None, type=int,
                    help="The subjects to process")
parser.add_argument('--timepoint', default=1,
                    help="The time point to process")
args = parser.parse_args()

DOMAIN = 'system'
USER = 'manager'
SRC_PROJECT = 71
DEST_PROJECT = 88
REPO_ID = 2


expected_patterns = {
    r'localizer': 'Localizer',
    r'ep2d_diff_mrtrix_33_dir_3_inter_b0_p2_RL': 'Diffusion',
    r'PRE DWI L-R distortion correction 36 DIR MRTrix': 'Diffusion_DISTCOR',
    r'REST_cmrr_mbep2d_bold_mat64_32Ch': 'rsfMRI',
    r'REST_cmrr_mbep2d_bold_mat64_32Ch_REST_cmrr_mbep2d_bold_mat64_32Ch_SBRef':
        'rsfMRI_REF',
    r'T2swi3d_ axial_p2_1.8mm_SWI_Images': 'SWI',
    r'T2swi3d_ axial_p2_1.8mm_Mag_Images': 'SWI_Mag',
    r'T2swi3d_ axial_p2_1.8mm_mIP_Images(SW)': 'SWI_mIP',
    r'T2swi3d_ axial_p2_1.8mm_Pha_Images': 'SWI_Pha',
    # r'': 'SWI_CSENSE_RECON',  # k-space/raw data ??
    # r'': 'SWI_COILS',  # k-space/raw data ??
    # r'gre_field_mapping 3mm': 'FMAP_MAG_ECH1',
    # r'gre_field_mapping 3mm': 'FMAP_MAG_ECH2',
    # r'': 'FMAP_COILS',  # ??
    # r'': 'FMAP_PHA',
    r't1_mprage_sag_p2_iso_1_ADNI': 'MPRAGE',
    r't2_spc_da-fl_sag_p2_iso_1.0': 'FLAIR',
    r'tgse_pasl_m0': 'ASL_3D_GRASE_REF',
    r'tgse_pasl_m0_Perfusion_Weighted': 'ASL_3D_GRASE'}

try:
    temp_dir = tempfile.mkdtemp()

    with DarisLogin(domain=DOMAIN, user=USER,
                      password=args.password) as daris:
        if args.subject_ids is None:
            subject_ids = daris.get_subjects(project_id=SRC_PROJECT,
                                             repo_id=REPO_ID).keys()
        else:
            subject_ids = args.subject_ids
        for subject_id in subject_ids:
            files = daris.get_files(
                session_id=args.timepoint, subject_id=subject_id,
                project_id=SRC_PROJECT, repo_id=REPO_ID)
            new_names = {}
            for pattern in expected_patterns:
                matched_file = None
                for f in files:
                    if re.match(pattern, f.name) is not None:
                        if matched_file is not None:
                            raise Exception(
                                "Multiple files match pattern '{}', {} and {}"
                                .format(pattern, matched_file, f))
                        matched_file = f
                if matched_file is None:
                    raise Exception(
                        "No files match the pattern '{}'"
                        .format(pattern, matched_file, f))
                new_names[expected_patterns[pattern]] = matched_file
            session_id = daris.add_session(
                subject_id=subject_id, ex_method_id=1, project_id=DEST_PROJECT,
                repo_id=REPO_ID)
            for name, f in new_names.iteritems():
                src_path = os.path.join(temp_dir, f.name)
                dest_path = os.path.join(temp_dir, name + '.nii.gz')
                daris.download(src_path, file_id=f.id, subject_id=subject_id,
                               project_id=SRC_PROJECT, repo_id=REPO_ID,
                               ex_method_id=1)
                sp.check_call(
                    'mrconvert {} {}'.format(src_path, dest_path))
                daris.add_file(subject_id=subject_id, ex_method_id=1,
                               session_id=session_id, project_id=DEST_PROJECT,
                               repo_id=REPO_ID)
                daris.upload(dest_path, subject_id=subject_id, ex_method_id=1,
                             session_id=session_id, project_id=DEST_PROJECT,
                             repo_id=REPO_ID)
finally:
    shutil.rmtree(temp_dir)
