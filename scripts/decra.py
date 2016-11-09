#!/usr/bin/env python
"""
Plots a workflow graph using NiPype's write_graph function and then loads and
plots the resulting PNG
"""
from argparse import ArgumentParser
from nianalysis.base import Dataset
from nianalysis import LocalArchive, DiffusionProject
from nianalysis.formats import nifti_gz_format

parser = ArgumentParser()
# parser.add_argument('--subjects', type=str, nargs='+', default=None,
#                     help=("Subject ids over which to run the diffusion "
#                           "processing"))
# parser.add_argument('output_dir', type=str,
#                     help=("Output directory where FA, CSD and tracks datasets"
#                           " are stored."))
# parser.add_argument('--working_dir', type=str, default=None,
#                     help=("The directory where the intermediate datasets are "
#                           "stored"))
args = parser.parse_args()

archive = LocalArchive('/Users/tclose/Data/MBI/decra/local_archive')

project = DiffusionProject(
    'DecraDiffusion', project_id='143', archive=archive,
    input_scans={'dwi_scan': Dataset('14', format=nifti_gz_format),
                 'forward_rpe': Dataset('13', format=nifti_gz_format),
                 'reverse_rpe': Dataset('12', format=nifti_gz_format)})

project.fod_pipeline().run()
