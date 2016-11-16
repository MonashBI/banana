#!/usr/bin/env python
"""
Plots a workflow graph using NiPype's write_graph function and then loads and
plots the resulting PNG
"""
import os
from argparse import ArgumentParser
from nianalysis.archive.daris import DarisArchive
from nianalysis.study.mri.diffusion import DiffusionStudy

parser = ArgumentParser()
parser.add_argument('--subjects', type=str, nargs='+', default=None,
                    help=("Subject ids over which to run the diffusion "
                          "processing"))
parser.add_argument('output_dir', type=str,
                    help=("Output directory where FA, CSD and tracks datasets"
                          " are stored."))
parser.add_argument('--working_dir', type=str, default=None,
                    help=("The directory where the intermediate datasets are "
                          "stored"))
args = parser.parse_args()

study = DiffusionStudy(
    DarisArchive('tclose', os.environ['DARIS_PASSWORD']), 'Diffusion')
for subject_id in args.subjects:
    study.process_subject('brain_extraction', subject_id)
