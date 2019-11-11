#!/usr/bin/env python
import os.path
import shutil
from arcana.data import FilesetFilter
from banana.analysis.mri.diffusion import DwiAnalysis
from arcana.repository.basic import LocalFileSystemRepo
from banana.file_format import mrtrix_image_format

repository_path = os.path.abspath(os.path.join(
    os.environ['HOME'], 'Data', 'MBI', 'tbi', ))
BASE_WORK_PATH = os.path.abspath(os.path.join(
    os.environ['HOME'], 'Data', 'MBI', 'work'))

WORK_PATH = os.path.abspath(os.path.join(BASE_WORK_PATH, 'tbi_test'))
DATASET_NAME = 'tbi_test'

shutil.rmtree(WORK_PATH, ignore_errors=True)
os.makedirs(WORK_PATH)
analysis = DwiAnalysis(
    name=DATASET_NAME,
    project_id='2_vs_2.5',
    repository=LocalFileSystemRepo(repository_path),
    input_scans={
        'dwi_scan': Fileset('R-L_60dir_b2000', mrtrix_image_format),
        'forward_rpe': Fileset('R-L_6dir_b0', mrtrix_image_format),
        'reverse_rpe': Fileset('L-R_6dir_b0', mrtrix_image_format)})
analysis.bias_correct_pipeline(bias_method='fsl').run(work_dir=WORK_PATH)
analysis.fod_pipeline().run(work_dir=WORK_PATH)
analysis.fa_pipeline().run(work_dir=WORK_PATH)
