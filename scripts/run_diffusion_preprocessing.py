#!/usr/bin/env python
import os.path
import errno
from nianalysis.study.mri.structural.diffusion import DiffusionStudy
from arcana.repository.xnat import XnatRepository
from nianalysis.file_format import dicom_format
import logging
import argparse
from arcana.dataset.match import DatasetMatch
from arcana.runner.linear import LinearRunner

parser = argparse.ArgumentParser()
parser.add_argument('--subjects', nargs='+', type=int, default=None,
                    help="Subject IDs to process")
args = parser.parse_args()

scratch_dir = os.path.expanduser('~/scratch')

WORK_PATH = os.path.join(scratch_dir, 'mnd', 'diffusion')
try:
    os.makedirs(WORK_PATH)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

logger = logging.getLogger('NiAnalysis')
logger.setLevel(logging.DEBUG)
# Stream Handler
handler = logging.StreamHandler()
formatter = logging.Formatter("%(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
# File Handler
handler = logging.FileHandler(os.path.join(WORK_PATH, 'out.log'))
formatter = logging.Formatter("%(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

study = DiffusionStudy(name='diffusion', repository=XnatRepository(
    project_id='MRH060', server='https://mbi-xnat.erc.monash.edu.au',
    cache_dir=os.path.join(scratch_dir, 'xnat_cache-mnd')),
                       runner=LinearRunner(
                                           work_dir=os.path.join(scratch_dir, 'xnat_working_dir-mnd')),
                       inputs=[
                               DatasetMatch(
                                            'primary', dicom_format,
                                            'R-L_MRtrix_60_directions_interleaved_B0_ep2d_diff_p2'),
                               DatasetMatch(
                                            'dwi_reference', dicom_format,
                                            'L-R_MRtrix_60_directions_interleaved_B0_ep2d_diff_p2')],
                       subject_ids=['C01'],
                       visit_ids=['MR02'],
                       parameters={'preproc_pe_dir': 'RL'})

fods = study.data('fod')
# print(fods[0].path)
print('Done')
