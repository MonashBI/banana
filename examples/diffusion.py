#!/usr/bin/env python3
import os.path as op
from arcana import DatasetMatch, XnatRepository, SlurmRunner
from nianalysis.study.mri.structural.diffusion import DiffusionStudy
from nianalysis.file_format import dicom_format

# Create study object that accesses MBI's XNAT and submits jobs to the SLURM scheduler
study = DiffusionStudy(
    name='example_diffusion',
    repository=XnatRepository(
        project_id='MRH017',
        server='https://mbi-xnat.erc.monash.edu.au',
        cache_dir=op.expanduser('~/cache')),
    runner=SlurmRunner(work_dir=op.expanduser('~/work')),
    inputs={'primary': DatasetMatch('R-L ep2d_diff.*', dicom_format, is_regex=True),
            'reverse_phase': DatasetMatch('L-R ep2d_diff.*', dicom_format, is_regex=True)},
    parameters={'num_wb_tracks': 1e8},
    switches={'toolchain': 'mrtrix'})

# Generate whole brain tracks and return path to cached dataset
wb_tracks = study.data('whole_brain_tracks')
print("Whole brain tracks are available at '{}'".format(wb_tracks.path))
