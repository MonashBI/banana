#!/usr/bin/env python
"""
Plots a workflow graph using NiPype's write_graph function and then loads and
plots the resulting PNG
"""
from argparse import ArgumentParser
from nianalysis.base import Scan
from nianalysis import LocalArchive, DiffusionDataset
from nianalysis.formats import nifti_gz_format

parser = ArgumentParser()
# parser.add_argument('--subjects', type=str, nargs='+', default=None,
#                     help=("Subject ids over which to run the diffusion "
#                           "processing"))
# parser.add_argument('output_dir', type=str,
#                     help=("Output directory where FA, CSD and tracks files"
#                           " are stored."))
# parser.add_argument('--working_dir', type=str, default=None,
#                     help=("The directory where the intermediate files are "
#                           "stored"))
args = parser.parse_args()

archive = LocalArchive('/Users/tclose/Data/MBI/decra/local_archive')

dataset = DiffusionDataset(
    'DecraDiffusion', project_id='143', archive=archive,
    input_scans={'dwi_scan': Scan('14', format=nifti_gz_format),
                 'forward_rpe': Scan('13', format=nifti_gz_format),
                 'reverse_rpe': Scan('12', format=nifti_gz_format)})

dataset.fod_pipeline().run()
