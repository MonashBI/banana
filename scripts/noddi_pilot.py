#!/usr/bin/env python
import os.path
import shutil
from nianalysis.dataset import Dataset
from mbianalysis.study.mri.diffusion import NODDIStudy
from nianalysis.archive.local import LocalArchive
from nianalysis.data_formats import mrtrix_format

archive_path = os.path.abspath(os.path.join(
    os.environ['HOME'], 'Data', 'MBI', 'noddi'))
BASE_WORK_PATH = os.path.abspath(os.path.join(
    os.environ['HOME'], 'Data', 'MBI', 'work'))

NODDI_PROJECT = 'pilot'
NODDI_SUBJECT = 'SUBJECT1'
NODDI_SESSION = 'SESSION1'
WORK_PATH = os.path.abspath(os.path.join(BASE_WORK_PATH, 'noddi'))
SESSION_DIR = os.path.join(archive_path, NODDI_PROJECT,
                           NODDI_SUBJECT, NODDI_SESSION)
DATASET_NAME = 'noddi'

shutil.rmtree(WORK_PATH, ignore_errors=True)
os.makedirs(WORK_PATH)
study = NODDIStudy(
    name=DATASET_NAME,
    project_id=NODDI_PROJECT, archive=LocalArchive(archive_path),
    input_scans={
        'low_b_dw_scan': Dataset(
            'r_l_noddi_b700_30_directions', mrtrix_format),
        'high_b_dw_scan': Dataset(
            'r_l_noddi_b2000_60_directions', mrtrix_format),
        'forward_rpe': Dataset('r_l_noddi_b0_6', mrtrix_format),
        'reverse_rpe': Dataset('l_r_noddi_b0_6', mrtrix_format)})
study.noddi_fitting_pipeline().run(work_dir=WORK_PATH)
