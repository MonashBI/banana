#!/usr/bin/env python
import os.path
import shutil
from arcana.data import InputFilesets
from banana.analysis.mri.diffusion import NODDIAnalysis
from arcana.repository.basic import BasicRepo
from banana.file_format import mrtrix_image_format

repository_path = os.path.abspath(os.path.join(
    os.environ['HOME'], 'Data', 'MBI', 'noddi'))
BASE_WORK_PATH = os.path.abspath(os.path.join(
    os.environ['HOME'], 'Data', 'MBI', 'work'))

NODDI_PROJECT = 'pilot'
NODDI_SUBJECT = 'SUBJECT1'
NODDI_SESSION = 'SESSION1'
WORK_PATH = os.path.abspath(os.path.join(BASE_WORK_PATH, 'noddi'))
SESSION_DIR = os.path.join(repository_path, NODDI_PROJECT,
                           NODDI_SUBJECT, NODDI_SESSION)
DATASET_NAME = 'noddi'

shutil.rmtree(WORK_PATH, ignore_errors=True)
os.makedirs(WORK_PATH)
analysis = NODDIAnalysis(
    name=DATASET_NAME,
    project_id=NODDI_PROJECT, repository=BasicRepo(repository_path),
    input_scans=[
        InputFilesets('low_b_dw_scan', mrtrix_image_format,
                     'r_l_noddi_b700_30_directions'),
        InputFilesets('high_b_dw_scan', mrtrix_image_format,
                     'r_l_noddi_b2000_60_directions'),
        InputFilesets('forward_rpe', 'r_l_noddi_b0_6', mrtrix_image_format),
        InputFilesets('reverse_rpe', 'l_r_noddi_b0_6', mrtrix_image_format)])
analysis.noddi_fitting_pipeline().run(work_dir=WORK_PATH)
