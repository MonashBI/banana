#!/usr/bin/env python
import os.path
import shutil
from nianalysis import Dataset, ASPREE_FSProject, LocalArchive
from nianalysis.formats import dicom_format

ARCHIVE_PATH = os.path.abspath(os.path.join(
    os.environ['HOME'], 'Data', 'MBI', 'ARCHIVE_ROOT'))
BASE_WORK_PATH = os.path.abspath(os.path.join(
    os.environ['HOME'], 'Data', 'MBI', 'work'))

ASPREE_FS_PROJECT = 'NEURO'
DATASET_NAME = 'freesurfer'
WORK_PATH = os.path.abspath(os.path.join(BASE_WORK_PATH, DATASET_NAME))

shutil.rmtree(WORK_PATH, ignore_errors=True)
os.makedirs(WORK_PATH)
project = NODDIProject(
    name=ASPREE_FS_PROJECT,
    project_id=ASPREE_FS_PROJECT, archive=LocalArchive(ARCHIVE_PATH),
    input_datasets={
        't1': Dataset('13_t1_mprage_sag_p2_iso_1_ADNI', dicom_format)})
project.freesurfer_pipeline().run(work_dir=WORK_PATH)
