import os
import os.path as op
import shutil
import json
from argparse import ArgumentParser
from arcana.repository import DirectoryRepository
from arcana.data import FilesetSelector, FieldSelector, Field, Fileset
from importlib import import_module

STUDY_NAME = 'DERIVED'

parser = ArgumentParser()
parser.add_argument('study_class',
                    help="The full path to the study class to test")
parser.add_argument('in_dir', help=("The path to the directory that the input "
                                    "data"))
parser.add_argument('out_dir', help="The path to the output directory")
parser.add_argument('work_dir', help="The work directory")
parser.add_argument('--parameter', '-p', metavar=('NAME', 'VALUE'),
                    nargs=2, action='append', default=(),
                    help="Parameter to set when initialising the study")
parser.add_argument('--skip', '-s', action='append',
                    help="Spec names to skip in the generation process")
args = parser.parse_args()


parts = args.study_class.split('.')
module_name = parts[:-1]
class_name = parts[-1]

parameters = dict(args.parameter)

module = import_module(module_name)
study_class = getattr(module, class_name)

ref_repo = DirectoryRepository(args.in_dir, depth=0)

in_paths = []
inputs = []
for fname in os.listdir(args.in_dir):
    path = op.join(ref_repo.root_dir, fname)
    in_paths.append(path)
    if fname == DirectoryRepository.FIELDS_FNAME:
        with open(path) as f:
            field_data = json.load(f)
        for name, value in field_data.items():
            field = Field(name, value)
            inputs.append(FieldSelector(field.name, field.name, field.dtype))
    else:
        fileset = Fileset.from_path(path)
        inputs.append(FilesetSelector(fileset.name, fileset.name,
                                      fileset.format))

study = study_class(
    STUDY_NAME,
    args.work_dir,
    inputs=inputs,
    parameters=parameters)

# Generate all derived data
for spec in study.data_specs():
    if spec.name not in args.skip:
        study.data(spec.name)

shutil.move(op.join(args.in_dir, STUDY_NAME), args.out_dir)

# Copy inputs to output reference dir
for path in in_paths:
    out_path = op.join(args.out_dir, op.basename(path))
    if op.isdir(path):
        shutil.copytree(path, out_path)
    else:
        shutil.copy(path, out_path)
