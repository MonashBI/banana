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

mri_tester = SystemTester(MriStudy, args.ref_data_dir, args.work_dir)

results = mri_tester.test_all()

errors = []
failures = []
successes = []
for result in results:
    if result.type == 'success':
        successes.append(result)
    elif result.type == 'failure':
        failures.append(result)
    elif result.type == 'error':
        errors.append(result)
    else:
        assert False

for error in errors:
    print(error)

for failure in failures:
    print(failure)

if not errors and not failures:
    print("{} tests ran successfully".format(len(successes)))
else:
    print("{} errors and {} failures".format(len(errors), len(failures)))
