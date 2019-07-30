#!/usr/bin/env python
from banana.study.multi.mrpet import create_motion_detection_class
import os
import os.path as op
import errno
from arcana.repository.basic import BasicRepo
from banana.utils.moco import (
    guess_scan_type, local_motion_detection, inputs_generation)
import argparse
import pickle as pkl
from arcana import SingleProc, ModulesEnv, StaticEnv
import logging


class MoCoDataLoader(object):

    def __init__(self, input_dir):

        self.input_dir = input_dir

    def load(self, pet_dir=None):

        cached_inputs = False
        cache_input_path = op.join(self.input_dir, 'inputs.pickle')

        if op.isdir(input_dir):
            try:
                with open(cache_input_path, 'rb') as f:
                    ref, ref_type, t1s, epis, t2s, dwis = pkl.load(f)
                cached_inputs = True
            except IOError as e:
                if e.errno == errno.ENOENT:
                    print('No inputs.pickle files found in {}. Running inputs'
                          ' generation'.format(self.input_dir))
        if not cached_inputs:
            scans = local_motion_detection(self.input_dir, pet_dir=pet_dir)

            list_inputs = guess_scan_type(scans, self.input_dir)

            if not list_inputs:
                ref, ref_type, t1s, epis, t2s, dwis = inputs_generation(
                    scans, self.input_dir, siemens=True)
                list_inputs = [ref, ref_type, t1s, epis, t2s, dwis]
            else:
                ref, ref_type, t1s, epis, t2s, dwis = list_inputs

            with open(cache_input_path, 'wb') as f:
                pkl.dump(list_inputs, f)

        return ref, ref_type, t1s, epis, t2s, dwis


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--input_dir', '-i', type=str,
                        help=("Path to an existing directory"))
    # parser.add_argument('--output_dir', '-o', type=str, default=None,
    #                     help="The output directory")
    parser.add_argument('--environment', choices=('static', 'modules'),
                        default='static',
                        help="The environment to use, modules or static")
    parser.add_argument('--reprocess', default=False, action='store_true',
                        help="reprocess the previously generated derivatives")
    args = parser.parse_args()
    input_dir = args.input_dir
    dataloader = MoCoDataLoader(input_dir)
    ref, ref_type, t1s, epis, t2s, dwis = dataloader.load()

    logger = logging.getLogger('nipype.workflow')

    logger.setLevel('INFO')
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    MotionDetection, inputs = create_motion_detection_class(
        'MotionDetection', ref, ref_type, t1s=t1s, t2s=t2s, dwis=dwis,
        epis=epis)

    sub_id = 'work_sub_dir'
    session_id = 'work_session_dir'
    # if args.output_dir is None:
    #     output_dir = op.join(op.basename(input_dir), 'work_dir')
    # else:
    #     output_dir = args.output_dir
    # os.makedirs(output_dir, exist_ok=True)
    repository = BasicRepo(input_dir, depth=0)
    work_dir = op.join(op.dirname(input_dir), 'motion_detection_cache')
    WORK_PATH = work_dir
    try:
        os.makedirs(WORK_PATH)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    study = MotionDetection(name='MotionDetection',
                            processor=SingleProc(WORK_PATH,
                                                 reprocess=args.reprocess),
                            environment=(
                                ModulesEnv() if args.environment == 'modules'
                                else StaticEnv()),
                            repository=repository, inputs=inputs,
                            subject_ids=[sub_id], visit_ids=[session_id])
    study.data('motion_detection_output')

print('Done!')
