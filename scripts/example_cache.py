#!/usr/bin/env python
import os.path
import errno
from arcana.data import FilesetFilter
from banana.analysis.mri.t2star import T2starAnalysis
from arcana.repository.xnat import XnatRepo
from banana.file_format import zip_format
import argparse
import pickle as pkl

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
print(session_ids_path)
with open(session_ids_path) as f:
    ids = f.read().split()

PROJECT_ID = 'MRH017'
filesets = {FilesetFilter('coils', 'swi_coils', zip_format)}
visit_ids = visit_ids['MR01']

repository = XnatRepo(cache_dir='/scratch/dq13/xnat_cache3')

if args.cache_project:
    project = repository.project(PROJECT_ID, subject_ids=ids,
                                 visit_ids=visit_ids)
    with open(CACHE_PROJECT_PATH, 'w') as f:
        pkl.dump(project, f)
else:
    with open(CACHE_PROJECT_PATH) as f:
        project = pkl.load(f)   


repository.cache(PROJECT_ID, filesets.values(), subject_ids=ids,
                 visit_ids=visit_ids)
    
analysis = T2starAnalysis(
    name='qsm',
    project_id=PROJECT_ID, repository=repository, input_filesets=filesets)
analysis.qsm_pipeline().submit(subject_ids=ids, visit_ids=visit_ids,
                            work_dir=WORK_PATH, email='tom.close@monash.edu',
                            project=project)
