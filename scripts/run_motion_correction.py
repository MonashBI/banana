from nianalysis.study.multimodal.mrpet import create_motion_correction_class
import os.path
import errno
from arcana.runner import MultiProcRunner
from arcana.archive.local import LocalArchive
from mc_pipeline.utils import (guess_scan_type, local_motion_detection,
                               inputs_generation)
import argparse
import cPickle as pkl
from arcana.runner.linear import LinearRunner


class RunMotionCorrection:

    def __init__(self, input_dir, pet_dir=None, dynamic=False, bin_len=60,
                 pet_offset=0, frames='all', struct2align=None,
                 pet_recon=None):

        self.input_dir = input_dir
        self.pet_dir = pet_dir
        self.dynamic = dynamic
        self.struct2align = struct2align
        self.pet_recon = pet_recon
        self.options = {'fixed_binning_n_frames': frames,
                        'fixed_binning_pet_offset': pet_offset,
                        'fixed_binning_bin_len': bin_len}

    def create_motion_correction_inputs(self):

        input_dir = self.input_dir
        pet_dir = self.pet_dir
        pet_recon = self.pet_recon
        struct2align = self.struct2align
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
            scans = local_motion_detection(input_dir, pet_dir=pet_dir,
                                           pet_recon=pet_recon,
                                           struct2align=struct2align)
            list_inputs = guess_scan_type(scans, input_dir)
            if not list_inputs:
                ref, ref_type, t1s, epis, t2s, dmris, utes, umap = (
                    inputs_generation(scans, input_dir, siemens=False))
                list_inputs = [ref, ref_type, t1s, epis, t2s, dmris, utes,
                               umap]
            else:
                print list_inputs
                ref, ref_type, t1s, epis, t2s, dmris, utes, umap = list_inputs
            with open(cache_input_path, 'w') as f:
                pkl.dump(list_inputs, f)

        return ref, ref_type, t1s, epis, t2s, dmris, utes, umap


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--input_dir', '-i', type=str, required=True,
                        help=("Path to an existing directory"))
    parser.add_argument('--pet_list_mode_dir', '-ls', type=str,
                        help=("Path to an existing directory with the PET list"
                              "-mode data (both binary and header files)."))
    parser.add_argument('--pet_reconstructed_dir', '-recon', type=str,
                        help=("Path to an existing directory with the PET "
                              "reconstructed data (one folder containing DICOM"
                              " files per frame)."))
    parser.add_argument('--bin_length', '-l', type=int,
                        help=("If dynamic motion correction, the temporal "
                              "length of each bin has to be provided (in sec)."
                              " Here we assume that each bin has the same "
                              "temporal duration. Default is 60 seconds."))
    parser.add_argument('--recon_offset', '-po', type=int,
                        help=("If dynamic motion correction, this is the time "
                              "difference, in seconds, between the PET start "
                              "time and the start time of the first "
                              "reconstructed bin. Default is 0."))
    parser.add_argument('--frames', '-f', type=int,
                        help=("If dynamic motion correction, this is the "
                              "number of reconstructed frames that have to be"
                              "corrected for motion. Default is equal to the "
                              "total PET acquisition length divided by the bin"
                              " temporal length."))
    parser.add_argument('--dynamic', '-d', action='store_true',
                        help=("If provided, dynamic motion correction will be "
                              "performed. Otherwise static. Default is static."
                              ""), default=False)
    parser.add_argument('--struct2align', '-s', type=str,
                        help=("Existing nifti file to register the final "
                              "motion correction PET image to. Default is None"
                              "."), default=None)
    args = parser.parse_args()
#     input_dir = '/Volumes/Project/pet/sforazz/test_mc_nianalysis/MRH017_006/MR01/'
#     input_dir = args.input_dir
    mc = RunMotionCorrection(
        args.input_dir, pet_dir=args.pet_list_mode_dir, dynamic=args.dynamic,
        bin_len=args.bin_length, pet_offset=args.recon_offset,
        frames=args.frames, pet_recon=args.pet_reconstructed_dir,
        struct2align=args.struct2align)

    ref, ref_type, t1s, epis, t2s, dmris, utes, umap = (
        mc.create_motion_correction_inputs())

    MotionCorrection, inputs = create_motion_correction_class(
        'MotionDetection', ref, ref_type, t1s=t1s, t2s=t2s, dmris=dmris,
        epis=epis, utes=utes, umaps=umap, pet_data_dir=args.pet_list_mode_dir,
        pet_recon_dir=args.pet_reconstructed_dir, dynamic=args.dynamic,
        struct2align=args.struct2align)

    sub_id = 'work_sub_dir'
    session_id = 'work_session_dir'
    archive = LocalArchive(args.input_dir+'/work_dir')
    work_dir = os.path.join(args.input_dir, 'motion_detection_cache')
    WORK_PATH = work_dir
    try:
        os.makedirs(WORK_PATH)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    study = MotionCorrection(name='MotionCorrection',
                             runner=MultiProcRunner(WORK_PATH),
                             archive=archive, inputs=inputs,
                             subject_ids=[sub_id], options=mc.options,
                             visit_ids=[session_id])
    study.data('motion_detection_output')
print 'Done!'
