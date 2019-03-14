#!/usr/bin/env python3
import sys
import os.path as op
from arcana.repository import BasicRepo
from argparse import ArgumentParser
from importlib import import_module
sys.path.insert(0, op.dirname(__file__))
from framework import SystemTester  # @IgnorePep8 @UnresolvedImport

parser = ArgumentParser()
parser.add_argument('study_class',
                    help="The full path to the study class to test")
parser.add_argument('pipeline_name',
                    help="The name of the pipeline to test")
parser.add_argument('reference_dir',
                    help=("The path to the directory that contains the "
                          "reference data"))
parser.add_argument('--work_dir', '-w', default=None,
                    help=("The path of the working directory"))
parser.add_argument('--criteria', '-c', default='exact',
                    help="The criteria to use to run the test")
parser.add_argument('--parameter', '-p', metavar=('NAME', 'VALUE'),
                    nargs=2, action='append', default=(),
                    help="Parameter to set when initialising the study")
parser.add_argument('--dry_run', default=False, action='store_true',
                    help=("Create the tests but don't actually run the "
                          "pipeline to see if the tests are constructed "
                          "properly"))
args = parser.parse_args()

parts = args.study_class.split('.')
module_name = '.'.join(parts[:-1])
class_name = parts[-1]

parameters = dict(args.parameter)

module = import_module(module_name)
study_class = getattr(module, class_name)

ref_repo = BasicRepo(args.reference_dir, depth=0)

tester = SystemTester(study_class, ref_repo, args.work_dir,
                      parameters=parameters, dry_run=args.dry_run)
results = tester.test(args.pipeline_name, criteria=args.criteria)

SystemTester.print_results(results)
