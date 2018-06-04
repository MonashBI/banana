#!/usr/bin/env python
import os.path
import shutil
from arcana.dataset import DatasetMatch
from nianalysis.study.mri.diffusion import NODDIStudy
from arcana.repository.local import LocalRepository
from nianalysis.data_format import mrtrix_format

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
study = NODDIStudy(
    name=DATASET_NAME,
    project_id=NODDI_PROJECT, repository=LocalRepository(repository_path),
    input_scans=[
        DatasetMatch('low_b_dw_scan', mrtrix_format,
                     'r_l_noddi_b700_30_directions'),
        DatasetMatch('high_b_dw_scan', mrtrix_format,
                     'r_l_noddi_b2000_60_directions'),
        DatasetMatch('forward_rpe', mrtrix_format, 'r_l_noddi_b0_6'),
        DatasetMatch('reverse_rpe', mrtrix_format, 'l_r_noddi_b0_6')])
study.noddi_fitting_pipeline().run(work_dir=WORK_PATH)
