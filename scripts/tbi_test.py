#!/usr/bin/env python
import os.path
import shutil
from nianalysis import Scan, DiffusionDataset, LocalArchive
from nianalysis.file_formats import mrtrix_format

ARCHIVE_PATH = os.path.abspath(os.path.join(
    os.environ['HOME'], 'Data', 'MBI', 'tbi', ))
BASE_WORK_PATH = os.path.abspath(os.path.join(
    os.environ['HOME'], 'Data', 'MBI', 'work'))

WORK_PATH = os.path.abspath(os.path.join(BASE_WORK_PATH, 'tbi_test'))
DATASET_NAME = 'tbi_test'

shutil.rmtree(WORK_PATH, ignore_errors=True)
os.makedirs(WORK_PATH)
dataset = DiffusionDataset(
    name=DATASET_NAME,
    project_id='2_vs_2.5',
    archive=LocalArchive(ARCHIVE_PATH),
    input_scans={
        'dwi_scan': Scan('R-L_60dir_b2000', mrtrix_format),
        'forward_rpe': Scan('R-L_6dir_b0', mrtrix_format),
        'reverse_rpe': Scan('L-R_6dir_b0', mrtrix_format)})
dataset.bias_correct_pipeline(bias_method='fsl').run(work_dir=WORK_PATH)
dataset.fod_pipeline().run(work_dir=WORK_PATH)
dataset.fa_pipeline().run(work_dir=WORK_PATH)
