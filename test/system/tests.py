#!/usr/bin/env python3
import sys
import os.path as op
from argparse import ArgumentParser
from banana.study.mri import MriStudy
sys.path.insert(0, op.dirname(__file__))
from framework import SystemTester  # @IgnorePep8 @UnresolvedImport


parser = ArgumentParser()
parser.add_argument('ref_data_dir', type=str,
                    help="The directory containing the reference data")
parser.add_argument('--work_dir', type=str, default=None,
                    help="The working directory to use")
args = parser.parse_args()

results = []

mri_tester = SystemTester(MriStudy, args.ref_data_dir, args.work_dir)
results.extend(mri_tester.test_all())

SystemTester.print_results(results)
