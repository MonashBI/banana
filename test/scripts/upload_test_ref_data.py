#!/usr/bin/env python3
import os
import os.path as op
from copy import copy
import json
import xnat
from argparse import ArgumentParser
from arcana.repository import BasicRepo, XnatRepo
from arcana.processor import SingleProc
from arcana.data import FilesetInput, FieldInput, Field, Fileset
from importlib import import_module
import tempfile
from arcana.exceptions import ArcanaNameError
import banana.file_format  # @UnusedImport

parser = ArgumentParser()
parser.add_argument('study_class',
                    help="The full path to the study class to test")
parser.add_argument('in_dir', help=("The path to the directory that the input "
                                    "data"))
parser.add_argument('project_id',
                    help=("The project ID, remaining is built from "
                          "the study class name"))
parser.add_argument('--server', default='https://mbi-xnat.erc.monash.edu.au',
                    help="The server to upload the reference data to")
parser.add_argument('--work_dir', default=None, help="The work directory")
parser.add_argument('--parameter', '-p', metavar=('NAME', 'VALUE'),
                    nargs=2, action='append', default=[],
                    help="Parameter to set when initialising the study")
parser.add_argument('--skip', '-s', nargs='+', default=[],
                    help="Spec names to skip in the generation process")
parser.add_argument('--reprocess', action='store_true', default=False,
                    help="Whether to reprocess the generated datasets")
parser.add_argument('--overwrite', default=False,
                    action='store_true', help="Overwrite existing data")
parser.add_argument('--repo_depth', type=int, default=0,
                    help="The depth of the input repository")
args = parser.parse_args()

if args.work_dir is None:
    work_dir = tempfile.mkdtemp()
else:
    work_dir = args.work_dir

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

if class_name.endswith('Study'):
    study_name = class_name[:-len('Study')]
else:
    study_name = class_name

ref_repo = BasicRepo(args.in_dir, depth=args.repo_depth)

in_paths = []
inputs = []
for fname in os.listdir(args.in_dir):
    if fname == study_name:
        continue
    path = op.join(ref_repo.root_dir, fname)
    in_paths.append(path)
    if fname == BasicRepo.FIELDS_FNAME:
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
    study_name,
    repository=ref_repo,
    processor=SingleProc(work_dir, reprocess=args.reprocess),
    inputs=inputs,
    parameters=parameters)

# Generate all derived data
for spec in study.data_specs():
    if spec.name not in args.skip:
        study.data(spec.name)

# Clear out existing data from project
if args.overwrite:
    with xnat.connect(args.server) as xlogin:
        for xsubject in xlogin.projects[args.project_id].subjects.values():
            for xsession in xsubject.experiments.values():
                for xscan in xsession.scans.values():
                    for xresource in xscan.resources.values():
                        xresource.delete()
                    xscan.delete()
                xsession.delete()
            xsubject.delete()

# Upload data to repository
out_repo = XnatRepo(args.server, project_id=args.project_id,
                    cache_dir=op.join(work_dir, 'xnat-cache'))
for spec in study.data_specs():
    if spec.name not in args.skip:
        for item in study.data(spec.name):
            if not item.exists:
                continue
            if item.is_fileset:
                item_cpy = Fileset(
                    name=item.name, format=item.format,
                    frequency=item.frequency, path=item.path,
                    side_cars=copy(item.side_cars),
                    subject_id=item.subject_id, visit_id=item.visit_id,
                    repository=out_repo, exists=True)
            else:
                item_cpy = Field(
                    name=item.name, value=item.value, dtype=item.dtype,
                    frequency=item.frequency, array=item.array,
                    subject_id=item.subject_id, visit_id=item.visit_id,
                    repository=out_repo, exists=True)
            item_cpy.put()
