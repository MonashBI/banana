#!/usr/bin/env python
import os.path
import shutil
from nianalysis import Dataset
from nianalysis.study.mri import MRStudy
from nianalysis.archive.local import LocalArchive
from nianalysis.data_formats import nifti_gz_format

ARCHIVE_PATH = os.path.abspath(os.path.join(
    os.environ['HOME'], 'Data', 'MBI', 'ARCHIVE_ROOT'))
BASE_WORK_PATH = os.path.abspath(os.path.join(
    os.environ['HOME'], 'Data', 'MBI', 'work'))

PROJECT = 'BIANCA'
DATASET_NAME = 'freesurfer'
WORK_PATH = os.path.abspath(os.path.join(BASE_WORK_PATH, DATASET_NAME))

shutil.rmtree(WORK_PATH, ignore_errors=True)
os.makedirs(WORK_PATH)
study = MRStudy(
    name='bianca',
    project_id=PROJECT, archive=LocalArchive(ARCHIVE_PATH),
    input_datasets={
        'mri_scan': Dataset('flair', nifti_gz_format)})
study.brain_mask_pipeline().run(work_dir=WORK_PATH)
