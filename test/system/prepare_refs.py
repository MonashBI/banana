#!/usr/bin/env python3
import os
import os.path as op
import shutil
import json
from argparse import ArgumentParser
from arcana.repository import DirectoryRepo
from arcana.processor import LinearProcessor
from arcana.data import FilesetInput, FieldInput, Field, Fileset
from importlib import import_module
from arcana.exceptions import ArcanaUsageError, ArcanaNameError
import banana.file_format  # @UnusedImport

STUDY_NAME = 'DERIVED'

parser = ArgumentParser()
parser.add_argument('study_class',
                    help="The full path to the study class to test")
parser.add_argument('in_dir', help=("The path to the directory that the input "
                                    "data"))
parser.add_argument('out_dir', help="The path to the output directory")
parser.add_argument('work_dir', help="The work directory")
parser.add_argument('--parameter', '-p', metavar=('NAME', 'VALUE'),
                    nargs=2, action='append', default=[],
                    help="Parameter to set when initialising the study")
parser.add_argument('--skip', '-s', nargs='+', default=[],
                    help="Spec names to skip in the generation process")
parser.add_argument('--reprocess', action='store_true', default=False,
                    help="Whether to reprocess the generated datasets")
args = parser.parse_args()


parts = args.study_class.split('.')
module_name = '.'.join(parts[:-1])
class_name = parts[-1]

parameters = {}
for name, value in args.parameter:
    try:
        value = int(value)
    except ValueError:
        try:
            value = float(value)
        except ValueError:
            pass
    parameters[name] = value

module = import_module(module_name)
study_class = getattr(module, class_name)

ref_repo = DirectoryRepo(args.in_dir, depth=0)

in_paths = []
inputs = []
for fname in os.listdir(args.in_dir):
    if fname == STUDY_NAME:
        continue
    path = op.join(ref_repo.root_dir, fname)
    in_paths.append(path)
    if fname == DirectoryRepo.FIELDS_FNAME:
        with open(path) as f:
            field_data = json.load(f)
        for name, value in field_data.items():
            field = Field(name, value)
            selector = FieldInput(field.name, field.name, field.dtype)
            inputs.append(selector)
    else:
        fileset = Fileset.from_path(path)
        selector = FilesetInput(fileset.basename, fileset.basename,
                                   fileset.format)
        try:
            spec = study_class.data_spec(selector)
        except ArcanaNameError:
            print("Skipping '{}' as it doesn't match a spec in {}"
                  .format(fname, args.study_class))
            continue
        if not spec.derived and selector.format not in spec.valid_formats:
            print("Skipping '{}' as it doesn't have a valid format for {} in "
                  " {} ({})".format(fname, spec.name, args.study_class,
                                    ', '.join(f.name
                                              for f in spec.valid_formats)))
            continue
        inputs.append(selector)

study = study_class(
    STUDY_NAME,
    repository=ref_repo,
    processor=LinearProcessor(args.work_dir, reprocess=args.reprocess),
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
