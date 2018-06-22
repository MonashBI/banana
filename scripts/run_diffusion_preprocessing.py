#!/usr/bin/env python3
import os.path
import errno
from nianalysis.study.mri.structural.diffusion import DiffusionStudy
from arcana.repository.xnat import XnatRepository
from nianalysis.file_format import dicom_format
import logging
import argparse
from arcana.dataset.match import DatasetMatch
from arcana.runner.linear import LinearRunner


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', type=str, nargs='+', default=None,
                        help="Subject IDs to process")
    parser.add_argument('--session', type=str, nargs='+', default=None,
                        help="Session IDs to process")
    parser.add_argument('--study_name', type=str, default='diffusion',
                        help="Study name to be prepend to the output names "
                        "of all pre-processing results. Default is "
                        "'diffusion'.")
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

    inputs = [
        DatasetMatch('primary', dicom_format,
                     'R-L_MRtrix_60_directions_interleaved_B0_ep2d_diff_p2'),
        DatasetMatch('dwi_reference', dicom_format,
                     'L-R_MRtrix_60_directions_interleaved_B0_ep2d_diff_p2')]

    study = DiffusionStudy(
        name=args.study_name,
        repository=XnatRepository(
            project_id='MRH060', server='https://mbi-xnat.erc.monash.edu.au',
            cache_dir=os.path.join(scratch_dir, 'xnat_cache-mnd')),
        runner=LinearRunner(work_dir=os.path.join(scratch_dir,
                                                  'xnat_working_dir-mnd')),
        inputs=inputs, subject_ids=args.subject, visit_ids=args.session,
        parameters={'preproc_pe_dir': 'RL'},
        switches={'preproc_denoise': True})

    fods = study.data('fod')
    # print(fods[0].path)
    print('Done')
