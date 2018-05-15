from nianalysis.study.mri.motion_detection_mixin_new import (
    create_motion_detection_class)
import os.path
import errno
from arcana.runner import MultiProcRunner
from arcana.archive.local import LocalArchive
# from nianalysis.archive.xnat import XNATArchive
from mc_pipeline.utils import (guess_scan_type, local_motion_detection,
                               inputs_generation)
import argparse
import cPickle as pkl
from arcana.runner.linear import LinearRunner


class A(object):
    pass


class create_motion_detection:

    def __init__(self, input_dir):

        self.input_dir = input_dir

    def create_md(self, pet_dir=None):

        input_dir = self.input_dir
        cached_inputs = False
        cache_input_path = os.path.join(input_dir, 'inputs.pickle')
        if os.path.isdir(input_dir):
            try:
                with open(cache_input_path, 'r') as f:
                    ref, ref_type, t1s, epis, t2s, dmris = pkl.load(f)
                cached_inputs = True
            except IOError, e:
                if e.errno == errno.ENOENT:
                    print ('No inputs.pickle files found in {}. Running inputs'
                           ' generation'.format(input_dir))
        if not cached_inputs:
            scans = local_motion_detection(input_dir, pet_dir=pet_dir)
            list_inputs = guess_scan_type(scans, input_dir)
            if not list_inputs:
                ref, ref_type, t1s, epis, t2s, dmris = inputs_generation(
                    scans, siemens=True)
                list_inputs = [ref, ref_type, t1s, epis, t2s, dmris]
            else:
                print list_inputs
                ref, ref_type, t1s, epis, t2s, dmris = list_inputs
            with open(cache_input_path, 'w') as f:
                pkl.dump(list_inputs, f)

        cached_inputs = False
        cache_input_path = os.path.join(input_dir, 'inputs.pickle')
        if os.path.isdir(input_dir):
            try:
                with open(cache_input_path, 'r') as f:
                    ref, ref_type, t1s, epis, t2s, dmris = pkl.load(f)
                cached_inputs = True
            except IOError, e:
                if e.errno == errno.ENOENT:
                    print ('No inputs.pickle files found in {}. Running inputs'
                           ' generation'.format(input_dir))
        if not cached_inputs:
            scans = local_motion_detection(input_dir, pet_dir=pet_dir)
            list_inputs = guess_scan_type(scans, input_dir)
            if not list_inputs:
                ref, ref_type, t1s, epis, t2s, dmris = inputs_generation(
                    scans, siemens=True)
                list_inputs = [ref, ref_type, t1s, epis, t2s, dmris]
            else:
                print list_inputs
                ref, ref_type, t1s, epis, t2s, dmris = list_inputs
            with open(cache_input_path, 'w') as f:
                pkl.dump(list_inputs, f)

        return ref, ref_type, t1s, epis, t2s, dmris


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--input_dir', '-i', type=str,
                        help=("Path to an existing directory"))
    args = parser.parse_args()
#     input_dir = '/Volumes/Project/pet/sforazz/MMH008_HD012_MRPT01/'
    input_dir = args.input_dir
    md = create_motion_detection(input_dir)
    ref, ref_type, t1s, epis, t2s, dmris = md.create_md()

    MotionDetection, inputs = create_motion_detection_class(
        'MotionDetection', ref, ref_type, t1s=t1s, t2s=t2s, dmris=dmris,
        epis=epis)

    MotionDetection.__module__ = A.__module__

    sub_id = 'work_sub_dir'
    session_id = 'work_session_dir'
    archive = LocalArchive(input_dir+'/work_dir')
    work_dir = os.path.join(input_dir, 'motion_detection_cache')
    WORK_PATH = work_dir
    try:
        os.makedirs(WORK_PATH)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    study = MotionDetection(name='MotionDetection',
                            runner=LinearRunner(WORK_PATH), archive=archive,
                            inputs=inputs, subject_ids=[sub_id],
                            visit_ids=[session_id])
    study.data('moco_series')
print 'Done!'
