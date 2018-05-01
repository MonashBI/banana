#!/usr/bin/env python
import os.path
import errno
from nianalysis.dataset import DatasetMatch
from mbianalysis.study.mri.structural.t2star import T2StarStudy
from nianalysis.archive.xnat import XNATArchive
from mbianalysis.data_format import zip_format
import argparse
import cPickle as pkl

parser = argparse.ArgumentParser()
parser.add_argument('--cache_project', help="Cache project to file", action='store_true',
                    default=False)
args = parser.parse_args()

WORK_PATH = os.path.join('/scratch', 'dq13', 'aspree', 'qsm')
CACHE_PROJECT_PATH = os.path.join(WORK_PATH, 'project.pkl')
try:
    os.makedirs(WORK_PATH)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise
session_ids_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), '..', 'resources',
    'old_swi_coils_remaining.txt')
print session_ids_path
with open(session_ids_path) as f:
    ids = f.read().split()

PROJECT_ID = 'MRH017'
datasets = {DatasetMatch('coils', zip_format, 'swi_coils')}
visit_ids = visit_ids['MR01']

archive = XNATArchive(cache_dir='/scratch/dq13/xnat_cache3')

if args.cache_project:
    project = archive.project(PROJECT_ID, subject_ids=ids, visit_ids=visit_ids)
    with open(CACHE_PROJECT_PATH, 'w') as f:
        pkl.dump(project, f)
else:
    with open(CACHE_PROJECT_PATH) as f:
        project = pkl.load(f)   


archive.cache(PROJECT_ID, datasets.values(), subject_ids=ids, visit_ids=visit_ids)
    
study = T2StarStudy(
    name='qsm',
    project_id=PROJECT_ID, archive=archive, input_datasets=datasets)
study.qsm_pipeline().submit(subject_ids=ids, visit_ids=visit_ids,
                            work_dir=WORK_PATH, email='tom.close@monash.edu',
                            project=project)
