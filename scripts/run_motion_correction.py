#!/usr/bin/env python3
from nianalysis.study.multimodal.mrpet import create_motion_correction_class
import os.path
import errno
# from arcana.runner import MultiProcRunner
from arcana.repository.local import LocalRepository
from nianalysis.motion_correction_utils import (
    guess_scan_type, local_motion_detection, inputs_generation)
import argparse
import pickle as pkl
from arcana.runner.linear import LinearRunner
import shutil


class RunMotionCorrection:

    def __init__(self, input_dir, pet_dir=None, dynamic=False, bin_len=60,
                 pet_offset=0, frames='all', struct2align=None,
                 pet_recon=None, crop_coordinates=None, mni_reg=False,
                 crop_size=None, static_len=0, pct_umap=False):

        self.input_dir = input_dir
        self.pet_dir = pet_dir
        self.dynamic = dynamic
        self.struct2align = struct2align
        self.pet_recon = pet_recon
        self.parameters = {
            'fixed_binning_n_frames': frames,
            'pet_offset': pet_offset,
            'fixed_binning_bin_len': bin_len,
            'PET2MNI_reg': mni_reg,
            'dynamic_pet_mc': dynamic,
            'framing_duration': static_len,
            'align_pct': pct_umap}
        if crop_coordinates is not None:
            crop_axes = ['x', 'y', 'z']
            for i, c in enumerate(crop_coordinates):
                self.parameters['crop_{}min'.format(crop_axes[i])] = c
        if crop_size is not None:
            crop_axes = ['x', 'y', 'z']
            for i, c in enumerate(crop_size):
                self.parameters['crop_{}size'.format(crop_axes[i])] = c

    def create_motion_correction_inputs(self):

        input_dir = self.input_dir
        pet_dir = self.pet_dir
        pet_recon = self.pet_recon
        struct2align = self.struct2align
        cached_inputs = False
        cache_input_path = os.path.join(input_dir, 'inputs.pickle')
        if os.path.isdir(input_dir):
            try:
                with open(cache_input_path, 'rb') as f:
                    (ref, ref_type, t1s, epis, t2s, dmris, pd,
                     pr) = pkl.load(f)
                working_dir = (
                    input_dir+'/work_dir/work_sub_dir/work_session_dir/')
                if pet_dir is not None and not pd and pd != pet_dir:
                    shutil.copytree(pet_dir, working_dir+'/pet_data_dir')
                if pet_recon is not None and pr != pet_recon:
                    if pr:
                        print('Different PET recon dir, respect to that '
                              'provided in a previous run. The directory '
                              'pet_data_reconstructed in the working directory'
                              ' will be removed and substituted with the new '
                              'one.')
                        shutil.rmtree(working_dir+'/pet_data_reconstructed')
                    shutil.copytree(pet_recon, working_dir +
                                    '/pet_data_reconstructed')
                    list_inputs = [ref, ref_type, t1s, epis, t2s, dmris, pd,
                                   pet_recon]
                    with open(cache_input_path, 'wb') as f:
                        pkl.dump(list_inputs, f)
                cached_inputs = True
            except IOError as e:
                if e.errno == errno.ENOENT:
                    print('No inputs.pickle files found in {}. Running inputs'
                          ' generation'.format(input_dir))
        if not cached_inputs:
            scans = local_motion_detection(input_dir, pet_dir=pet_dir,
                                           pet_recon=pet_recon,
                                           struct2align=struct2align)
            list_inputs = guess_scan_type(scans, input_dir)
            if not list_inputs:
                ref, ref_type, t1s, epis, t2s, dmris = (
                    inputs_generation(scans, input_dir, siemens=True))
                list_inputs = [ref, ref_type, t1s, epis, t2s, dmris]
            else:
                print(list_inputs)
                ref, ref_type, t1s, epis, t2s, dmris = (
                    list_inputs)
            if pet_dir is not None:
                list_inputs.append(pet_dir)
            else:
                list_inputs.append('')
            if pet_recon is not None:
                list_inputs.append(pet_recon)
            else:
                list_inputs.append('')
            with open(cache_input_path, 'wb') as f:
                pkl.dump(list_inputs, f)

        return ref, ref_type, t1s, epis, t2s, dmris


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--input_dir', '-i', type=str, required=True,
                        help=("Path to an existing directory"))
    parser.add_argument('--umap_ref', type=str,
                        help=("Path to the folder (within the input_dir) that "
                              "is the umap reference (usually UTE). This "
                              "will be used to realign the umap to match the "
                              "head position in each of the detected frames "
                              "This will be used ONLY for static motion "
                              "correction. Please see documentation for futher"
                              " explanation."), default=None)
    parser.add_argument('--umap', type=str,
                        help=("Path to the folder (within the input_dir) that "
                              "is the attenuation correction umap. This file "
                              "will be realigned to match the "
                              "head position in each detected frame during the"
                              " static motion correction. In order to "
                              "work, this file must be provided together with"
                              "--umap_ref."), default=None)
    parser.add_argument('--pet_list_mode_dir', '-ls', type=str,
                        help=("Path to an existing directory with the PET list"
                              "-mode data (both binary and header files)."))
    parser.add_argument('--pet_reconstructed_dir', '-recon', type=str,
                        help=("Path to an existing directory with the PET "
                              "reconstructed data (one folder containing DICOM"
                              " files per frame)."))
    parser.add_argument('--static_pet_len', '-sl', type=int,
                        help=("If static motion correction, this is the "
                              "length of PET data you want to reconstruct and "
                              "correct for motion. Default is from the "
                              "PET_start_time+recon_offset to the end of the "
                              "PET acquisition."), default=0)
    parser.add_argument('--bin_length', '-l', type=int,
                        help=("If dynamic motion correction, the temporal "
                              "length of each bin has to be provided (in sec)."
                              " Here we assume that each bin has the same "
                              "temporal duration. Default is 60 seconds."),
                        default=60)
    parser.add_argument('--recon_offset', '-ro', type=int,
                        help=("This is the time "
                              "difference, in seconds, between the PET start "
                              "time and the start time of the reconstruction "
                              "(valid for both dynamic and static motion "
                              "correction). Default is 0."), default=0)
    parser.add_argument('--frames', '-f', type=int,
                        help=("If dynamic motion correction, this is the "
                              "number of reconstructed frames that have to be"
                              "corrected for motion. Default is equal to the "
                              "total PET acquisition length divided by the bin"
                              " temporal length."), default=0)
    parser.add_argument('--dynamic', '-d', action='store_true',
                        help=("If provided, dynamic motion correction will be "
                              "performed. Otherwise static. Default is static."
                              ""), default=False)
    parser.add_argument('--struct2align', '-s', type=str,
                        help=("Existing nifti file to register ONLY the final "
                              "motion correction PET image to. This must be in"
                              " subject space and is highly recommended to be "
                              "brain extracted. Default is None"
                              "."), default=None)
    parser.add_argument('--cropping_coordinates', '-cc', type=int, nargs='+',
                        help=("x, y and z coordinates for cropping "
                              "the motion corrected PET image in the PET "
                              "space. Default is 100 100 20."), default=None)
    parser.add_argument('--cropping_size', '-cs', type=int, nargs='+',
                        help=("x, y and z size for the cropping, i.e. the "
                              "dimension along each of the axes. Default is "
                              "130 130 100"), default=None)
    parser.add_argument('--mni_reg', action='store_true',
                        help=("If provided, motion correction results will be "
                              "registered to PET template in MNI space. "
                              "Default is False."), default=False)
    parser.add_argument('--continuos_umap', action='store_true',
                        help=("If provided, the pipeline will assume that the"
                              "provided umap has continuous range of values "
                              "(for example pct umap). Otherwise discrete "
                              "(like UTE-based umap). Default is discrete."),
                        default=False)
    args = parser.parse_args()

    mc = RunMotionCorrection(
        args.input_dir, pet_dir=args.pet_list_mode_dir, dynamic=args.dynamic,
        bin_len=args.bin_length, pet_offset=args.recon_offset,
        frames=args.frames, pet_recon=args.pet_reconstructed_dir,
        struct2align=args.struct2align, crop_size=args.cropping_size,
        crop_coordinates=args.cropping_coordinates, mni_reg=args.mni_reg,
        static_len=args.static_pet_len, pct_umap=args.continuos_umap)

    ref, ref_type, t1s, epis, t2s, dmris = mc.create_motion_correction_inputs()

    MotionCorrection, inputs, out_data = create_motion_correction_class(
        'MotionCorrection', ref, ref_type, t1s=t1s, t2s=t2s, dmris=dmris,
        epis=epis, umap_ref=args.umap_ref, umap=args.umap,
        pet_data_dir=args.pet_list_mode_dir, dynamic=args.dynamic,
        pet_recon_dir=args.pet_reconstructed_dir,
        struct2align=args.struct2align)

    sub_id = 'work_sub_dir'
    session_id = 'work_session_dir'
    repository = LocalRepository(args.input_dir+'/work_dir')
    work_dir = os.path.join(args.input_dir, 'motion_detection_cache')
    WORK_PATH = work_dir
    try:
        os.makedirs(WORK_PATH)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    study = MotionCorrection(name='MotionCorrection',
                             runner=LinearRunner(WORK_PATH),
                             repository=repository, inputs=inputs,
                             subject_ids=[sub_id], parameters=mc.parameters,
                             visit_ids=[session_id])
    study.data(out_data)
print('Done!')
