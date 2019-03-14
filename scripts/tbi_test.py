#!/usr/bin/env python
import os.path
import shutil
from arcana.data import FilesetInput
from banana.study.mri.diffusion import DwiStudy
from arcana.repository.simple import DirectoryRepo
from banana.file_format import mrtrix_format

repository_path = os.path.abspath(os.path.join(
    os.environ['HOME'], 'Data', 'MBI', 'tbi', ))
BASE_WORK_PATH = os.path.abspath(os.path.join(
    os.environ['HOME'], 'Data', 'MBI', 'work'))

WORK_PATH = os.path.abspath(os.path.join(BASE_WORK_PATH, 'tbi_test'))
DATASET_NAME = 'tbi_test'

shutil.rmtree(WORK_PATH, ignore_errors=True)
os.makedirs(WORK_PATH)
study = DwiStudy(
    name=DATASET_NAME,
    project_id='2_vs_2.5',
    repository=DirectoryRepo(repository_path),
    input_scans={
        'dwi_scan': Fileset('R-L_60dir_b2000', mrtrix_format),
        'forward_rpe': Fileset('R-L_6dir_b0', mrtrix_format),
        'reverse_rpe': Fileset('L-R_6dir_b0', mrtrix_format)})
study.bias_correct_pipeline(bias_method='fsl').run(work_dir=WORK_PATH)
study.fod_pipeline().run(work_dir=WORK_PATH)
study.fa_pipeline().run(work_dir=WORK_PATH)
