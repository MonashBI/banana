#!/usr/bin/env python
import os.path
import shutil
from nianalysis import Dataset, DiffusionProject, LocalArchive
from nianalysis.formats import mrtrix_format

ARCHIVE_PATH = os.path.abspath(os.path.join(
    os.environ['HOME'], 'Data', 'MBI', 'tbi', ))
BASE_WORK_PATH = os.path.abspath(os.path.join(
    os.environ['HOME'], 'Data', 'MBI', 'work'))

WORK_PATH = os.path.abspath(os.path.join(BASE_WORK_PATH, 'tbi_test'))
DATASET_NAME = 'tbi_test'

shutil.rmtree(WORK_PATH, ignore_errors=True)
os.makedirs(WORK_PATH)
project = DiffusionProject(
    name=DATASET_NAME,
    project_id='2_vs_2.5',
    archive=LocalArchive(ARCHIVE_PATH),
    input_scans={
        'dwi_scan': Dataset('R-L_60dir_b2000', mrtrix_format),
        'forward_rpe': Dataset('R-L_6dir_b0', mrtrix_format),
        'reverse_rpe': Dataset('L-R_6dir_b0', mrtrix_format)})
project.bias_correct_pipeline(bias_method='fsl').run(work_dir=WORK_PATH)
project.fod_pipeline().run(work_dir=WORK_PATH)
project.fa_pipeline().run(work_dir=WORK_PATH)
