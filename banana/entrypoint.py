import sys
import os.path as op
import os
import subprocess as sp
import tempfile
from collections import defaultdict
import logging
import json
from pkgutil import iter_modules
from multiprocessing import cpu_count
from argparse import ArgumentParser
from importlib import import_module
from setuptools import find_packages
import neurodocker
from arcana.data import Fileset, Field
from arcana.utils import parse_value, wrap_text
from arcana.exceptions import ArcanaRequirementVersionsError
#from banana.utils.testing import AnalysisTester, PipelineTester
from banana.exceptions import BananaUsageError
from banana import (
    FilesetFilter, FieldFilter, MultiProc, SingleProc, SlurmProc, StaticEnv,
    ModulesEnv, BidsDataset, XnatRepo, XnatCSRepo, Analysis,
    MultiAnalysis, Dataset)
from banana.requirement import mrtrix_req, dcm2niix_req
from banana.__about__ import __version__

logger = logging.getLogger('banana')

DEFAULT_STUDY_CLASS_PATH = 'banana.analysis'

DEFAULT_LINE_LENGTH = 79
DEFAULT_INDENT = 4
DEFAULT_SPACER = 4


def set_loggers(loggers):

    # Overwrite earlier (default) versions of logger levels with later options
    loggers = dict(loggers)

    for name, level in loggers.items():
        logger = logging.getLogger(name)
        logger.setLevel(level)
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)


def resolve_class(class_str, prefixes=(DEFAULT_STUDY_CLASS_PATH,)):
    """
    Resolves a class from the '.' delimted module + class name string
    """
    parts = class_str.split('.')
    module_name = '.'.join(parts[:-1])
    class_name = parts[-1]
    cls = None
    for prefix in [None] + list(prefixes):
        if prefix is not None:
            mod_name = prefix + '.' + module_name
        else:
            mod_name = module_name
        if not mod_name:
            continue
        mod_name = mod_name.strip('.')
        try:
            module = import_module(mod_name)
        except ModuleNotFoundError:
            continue
        else:
            try:
                cls = getattr(module, class_name)
            except AttributeError:
                continue
            else:
                break
    if cls is None:
        raise BananaUsageError(
            "Did not find class '{}'".format(class_str))
    return cls


class DeriveCmd():

    desc = "Generate derivatives from a Banana Analysis class"

    @classmethod
    def parser(cls):
        parser = ArgumentParser(prog='banana derive',
                                description=cls.desc)
        parser.add_argument('dataset_name',
                            help=("Either the path to the dataset if of "
                                  "'bids' or 'basic' types, or the name of the"
                                  " project ID for 'xnat' type"))
        parser.add_argument('analysis_class',
                            help=("Name of the class to instantiate"))
        parser.add_argument('analysis_name',
                            help=("The name of the analysis to put the analysis "
                                  "under (e.g. parenthood)"))
        parser.add_argument('derivatives', nargs='+',
                            help=("The names of the derivatives to generate"))
        parser.add_argument('--repository', nargs='+', default=['bids'],
                            metavar='ARG',
                            help=("Specify the repository type and any options"
                                  " to be passed to it. First argument "))
        parser.add_argument('--output_dataset', '-o', nargs='+',
                            metavar='ARG', default=None,
                            help=("Specify a different output dataset "
                                  "to place the derivatives in. 1st arg "
                                  "is the type, one of ('basic', 'bids' or "
                                  "'xnat'). If type == 'xnat' then the "
                                  "following args are PROJECTID, SERVER, "
                                  "[USER, PASSWORD]"))
        parser.add_argument('--processor', default=['multi'], nargs='+',
                            metavar='ARG',
                            help=("The type of processor to use plus arguments"
                                  " used to initate it. First arg is the type "
                                  "(one of 'single', 'multi', 'slurm'). "
                                  "Additional arguments depend on type: "
                                  "single [], multi [NUM_PROCS], slurm ["
                                  "ACCOUNT, PARTITION]"))
        parser.add_argument('--environment', type=str, default='static',
                            choices=('modules', 'static'), metavar='TYPE',
                            help="The type of environment to use")
        parser.add_argument('--input', '-i', nargs='+', action='append',
                            default=[], metavar=('SPEC', 'PATTERN'),
                            help=("The inputs to include in the analysis init."
                                  " If not provided then all are used"))
        parser.add_argument('--parameter', '-p', nargs=2, action='append',
                            metavar=('NAME', 'VALUE'), default=[],
                            help="Parameters to pass to the analysis")
        parser.add_argument('--subject_ids', nargs='+', default=None,
                            metavar='ID',
                            help=("The subject IDs to include in the analysis."
                                  " If a single value with a '/' is "
                                  "provided then it is interpreted as a text "
                                  "file containing a list of IDs"))
        parser.add_argument('--visit_ids', nargs='+', default=None,
                            metavar='ID',
                            help=("The visit IDs to include in the analysis. "
                                  "If a single value with a '/' is "
                                  "provided then it is interpreted as a text "
                                  "file containing a list of IDs"))
        parser.add_argument('--session_ids', nargs='+', default=None,
                            help=("The session IDs to analyse. "
                                  "If a single value with a '/' is "
                                  "provided then it is interpreted as a text "
                                  "file containing a list of IDs. "
                                  "Note that session IDs doesn't restrict the "
                                  "breadth of analysis (like subject_ids and "
                                  "visit ids) just the sessions to be"
                                  "processed in this run, and if there are "
                                  "\"summary\" derivitives to be derived "
                                  "then upstream derivatives will be derived "
                                  "across the dataset "
                                  "(i.e. for other sessions as well)."))
        parser.add_argument('--scratch', type=str, default=None,
                            metavar='PATH',
                            help=("The scratch directory to use for the "
                                  "workflow and cache"))
        parser.add_argument('--cache', nargs='+', default=(), metavar='SPEC',
                            help=("Input filesets to cache locally before "
                                  "running workflows"))
        parser.add_argument('--enforce_inputs', action='store_true',
                            default=False,
                            help=("Whether to enforce inputs for non-optional "
                                  "specs"))
        parser.add_argument('--reprocess', action='store_true', default=False,
                            help=("Whether to reprocess previously generated "
                                  "derivatives with mismatching provenance"))
        parser.add_argument('--email', type=str, default=None,
                            help=("The email account to provide to SLURM "
                                  "scheduler"))
        parser.add_argument('--logger', nargs=2, action='append',
                            metavar=('LOGGER', 'LEVEL'),
                            default=[('banana', 'INFO'), ('arcana', 'INFO'),
                                     ('nipype.workflow', 'INFO')],
                            help=("Set levels for various loggers ('arcana', "
                                  "'banana', and 'nipype.workflow' are set to "
                                  "INFO by default)"))
        parser.add_argument('--quiet', action='store_true', default=False,
                            help=("Disable logging output"))
        parser.add_argument('--bids_task', default=None,
                            help=("A task to use to filter the BIDS inputs"))
        return parser

    @classmethod
    def run(cls, args):

        if not args.quiet:
            set_loggers(args.logger)

        analysis_class = resolve_class(args.analysis_class)

        if args.scratch is not None:
            scratch_dir = args.scratch
        else:
            scratch_dir = op.join(op.expanduser('~'), 'banana-scratch')

        # Ensure scratch dir exists
        os.makedirs(scratch_dir, exist_ok=True)

        work_dir = op.join(scratch_dir, 'work')

        if args.dataset is None:
            if args.input:
                dataset_type = 'basic'
            else:
                dataset_type = 'bids'
        else:
            dataset_type = args.dataset[0]

        # Load subject_ids from file if single value is provided with
        # a '/' in the string
        if (args.subject_ids is not None and args.subject_ids
                and '/' in args.subject_ids[0]):
            with open(args.subject_ids[0]) as f:
                subject_ids = f.read().split()
        else:
            subject_ids = args.subject_ids

        # Load visit_ids from file if single value is provided with
        # a '/' in the string
        if (args.visit_ids is not None and args.visit_ids
                and '/' in args.visit_ids[0]):
            with open(args.visit_ids[0]) as f:
                visit_ids = f.read().split()
        else:
            visit_ids = args.visit_ids

        # Load visit_ids from file if single value is provided with
        # a '/' in the string
        if (args.session_ids is not None and args.session_ids
                and '/' in args.session_ids[0]):
            with open(args.session_ids[0]) as f:
                session_ids = f.read().split()
        else:
            session_ids = args.visit_ids

        def init_dataset(dataset_path, dataset_type, option_str, dataset_args,
                         create_root=False, **kwargs):
            if dataset_type == 'bids':
                if create_root:
                    os.makedirs(dataset_path, exist_ok=True)
                dataset = BidsDataset(dataset_path, **kwargs)
            elif dataset_type == 'basic':
                if len(dataset_args) != 1:
                    raise BananaUsageError(
                        "Unrecognised arguments passed to '--{}' option "
                        "({}) exactly 1 additional argument is required for "
                        "'basic' type dataset (DEPTH)"
                        .format(option_str, dataset_args))
                if create_root:
                    os.makedirs(dataset_path, exist_ok=True)
                dataset = Dataset(dataset_path, depth=int(dataset_args[0]),
                                  **kwargs)
            elif dataset_type == 'xnat':
                nargs = len(dataset_args)
                if nargs < 1:
                    raise BananaUsageError(
                        "Not enough arguments passed to '--{}' option "
                        "({}), at least 1 additional argument is required for "
                        "'xnat' type dataset (SERVER)"
                        .format(option_str, dataset_args))
                elif nargs > 3:
                    raise BananaUsageError(
                        "Unrecognised arguments passed to '--{}' option "
                        "({}), at most 3 additional arguments are accepted for"
                        " 'xnat' type dataset (SERVER, USER, PASSWORD)"
                        .format(option_str, dataset_args))
                dataset = XnatRepo(
                    server=dataset_args[0],
                    user=(dataset_args[1] if nargs > 2 else None),
                    password=(dataset_args[2] if nargs > 3 else None),
                    cache_dir=op.join(scratch_dir,
                                      'cache')).dataset(dataset_path, **kwargs)
            else:
                raise BananaUsageError(
                    "Unrecognised dataset type provided as first argument "
                    "to '--{}' option ({})".format(option_str,
                                                   dataset_args[0]))
            return dataset

        dataset = init_dataset(args.dataset_path, dataset_type,
                               'dataset', args.dataset[1:],
                               subject_ids=subject_ids,
                               visit_ids=visit_ids)

        if args.output_dataset is not None:
            input_dataset = dataset
            tree = dataset.tree
            if subject_ids is None:
                subject_ids = list(tree.subject_ids)
            if visit_ids is None:
                visit_ids = list(tree.visit_ids)
            fill_tree = True
            nargs = len(args.output_dataset)
            if nargs == 1:
                dataset_type = 'basic'
                out_path = args.output_dataset[0]
                out_dataset_args = [input_dataset.depth]
            else:
                dataset_type = args.output_dataset[0]
                out_path = args.output_dataset[1]
                out_dataset_args = args.output_dataset[2:]
            dataset = init_dataset(out_path, dataset_type, 'output_dataset',
                                   out_dataset_args,
                                   create_root=True,
                                   subject_ids=subject_ids,
                                   visit_ids=visit_ids,
                                   fill_tree=fill_tree)
        else:
            input_dataset = None
            fill_tree = False

        if args.email is not None:
            email = args.email
        else:
            try:
                email = os.environ['EMAIL']
            except KeyError:
                email = None

        proc_args = {'reprocess': args.reprocess}

        if args.processor[0] == 'single':
            processor = SingleProc(work_dir, **proc_args)
        elif args.processor[0] == 'multi':
            if len(args.processor) > 1:
                num_processes = args.processor[1]
            elif len(args.processor) > 2:
                raise BananaUsageError(
                    "Unrecognised arguments passed to '--processor' option "
                    "({}) expected at most 1 additional argument for 'multi' "
                    "type processor (NUM_PROCS)".format(args.processor))
            else:
                num_processes = cpu_count()
            processor = MultiProc(work_dir, num_processes=num_processes,
                                  **proc_args)
        elif args.processor[0] == 'slurm':
            if email is None:
                raise BananaUsageError(
                    "Email needs to be provided either via '--email' argument "
                    "or set in 'EMAIL' environment variable for SLURM "
                    "processor")
            nargs = len(args.processor)
            if nargs > 3:
                raise BananaUsageError(
                    "Unrecognised arguments passed to '--processor' option "
                    "with 'slurm' type ({}), expected at most 2 additional "
                    "arguments [ACCOUNT, PARTITION]".format(args.processor))
            processor = SlurmProc(
                work_dir, account=(args.processor[1] if nargs >= 2 else None),
                partition=(args.processor[2] if nargs >= 3 else None),
                email=email, mail_on=('FAIL',),
                **proc_args)
        else:
            raise BananaUsageError(
                "Unrecognised processor type provided as first argument to "
                "'--processor' option ({})".format(args.processor[0]))

        if args.environment == 'static':
            environment = StaticEnv()
        else:
            environment = ModulesEnv()

        parameters = {}
        for name, value in args.parameter:
            parameters[name] = parse_value(
                value, dtype=analysis_class.param_spec(name).dtype)

        if input_dataset is not None and input_dataset.type == 'bids':
            inputs = analysis_class.get_bids_inputs(args.bids_task,
                                                    dataset=input_dataset)
        else:
            inputs = {}

        for inpt in args.input:
            name, pattern = inpt[:2]
            order = inpt[2] if (len(input) >= 3 and inpt[2] != '.') else None
            if len(inpt) > 3:
                dicom_tags = {}
                for arg in inpt[3:]:
                    try:
                        tag_name, val = arg.split('=')
                    except ValueError:
                        raise BananaUsageError(
                            "Header tags provided to '{}' input matcher need "
                            "to  in \"name=val\" format (not {})"
                            .format(name, arg))
                    dicom_tags[tag_name] = val
            else:
                dicom_tags = None
            spec = analysis_class.data_spec(name)
            if spec.is_fileset:
                inpt_cls = FilesetFilter
            else:
                inpt_cls = FieldFilter
            inputs[name] = inpt_cls(name, pattern=pattern, is_regex=True,
                                    dataset=input_dataset, order=order,
                                    dicom_tags=dicom_tags)

        analysis = analysis_class(
            name=args.analysis_name,
            dataset=dataset,
            processor=processor,
            environment=environment,
            inputs=inputs,
            parameters=parameters,
            enforce_inputs=args.enforce_inputs,
            bids_task=args.bids_task)

        for spec_name in args.cache:
            spec = analysis.bound_spec(spec_name)
            if not isinstance(spec, FilesetFilter):
                raise BananaUsageError(
                    "Cannot cache non-input fileset '{}'".format(spec_name))
            spec.cache()

        # Generate data
        analysis.derive(args.derivatives, session_ids=session_ids)

        logger.info("Generated derivatives for '%s'", args.derivatives)


# class TestGenCmd():

#     desc = ("Generate all derivatives from a analysis in a format compatible "
#             "with Banana's unit-testing framework")

#     @classmethod
#     def parser(cls):
#         parser = ArgumentParser(
#             prog='banana test-gen',
#             description=("Generates reference data for the built-in unittest "
#                          "framework given a analysis class, an input dataset "
#                          "containing data named according to the data "
#                          "specification of the class and set of parameters"))
#         parser.add_argument('analysis_class',
#                             help=("The path to the analysis class to test, e.g. "
#                                   "banana.analysis.MriAnalysis"))
#         parser.add_argument('in_dataset', help=("The path to dataset that "
#                                              "houses the input data"))
#         parser.add_argument('out_dataset',
#                             help=("If the 'xnat_server' argument is provided "
#                                   "then out is interpreted as the project ID "
#                                   "to use the XNAT server (the project must "
#                                   "exist already). Otherwise it is interpreted"
#                                   " as the path to a basic dataset"))
#         parser.add_argument('--in_server', default=None,
#                             help="The server to download the input data from")
#         parser.add_argument('--out_server', default=None,
#                             help="The server to upload the reference data to")
#         parser.add_argument('--work_dir', default=None,
#                             help="The work directory")
#         parser.add_argument('--parameter', '-p', metavar=('NAME', 'VALUE'),
#                             nargs=2, action='append', default=[],
#                             help=("Parameters to set when initialising the "
#                                   "analysis"))
#         parser.add_argument('--include', '-i', nargs='+', default=[],
#                             help=("Spec names to include in the generation "
#                                   "process. If not provided all (except "
#                                   "those that are explicitly skipped) "
#                                   "are included"))
#         parser.add_argument('--skip', '-s', nargs='+', default=[],
#                             help=("Spec names to skip in the generation "
#                                   "process"))
#         parser.add_argument('--bases', nargs='+', default=[],
#                             help=("Base classes which to include data specs "
#                                   "defined within them"))
#         parser.add_argument('--reprocess', action='store_true', default=False,
#                             help=("Whether to reprocess previously generated "
#                                   "datasets in the output dataset"))
#         parser.add_argument('--dataset_depth', type=int, default=0,
#                             help="The depth of the input dataset")
#         parser.add_argument('--dont_clean_work_dir', action='store_true',
#                             default=False,
#                             help=("Whether to clean the Nipype work dir "
#                                   "between runs"))
#         parser.add_argument('--loggers', nargs='+',
#                             default=('nipype.workflow', 'arcana', 'banana'),
#                             help="Loggers to set handlers to stdout for")
#         parser.add_argument('--environment', type=str, default='static',
#                             choices=('modules', 'static'), metavar='TYPE',
#                             help="The type of environment to use")
#         return parser

#     @classmethod
#     def run(cls, args):

#         # Get Analysis class
#         analysis_class = resolve_class(args.analysis_class)

#         include_bases = [resolve_class(c) for c in args.bases]

#         # Convert parameters to dictionary
#         parameters_dct = {}
#         for name, value in args.parameter:
#             try:
#                 value = int(value)
#             except ValueError:
#                 try:
#                     value = float(value)
#                 except ValueError:
#                     pass
#             parameters_dct[name] = value
#         parameters = parameters_dct

#         PipelineTester.generate_test_data(
#             analysis_class=analysis_class, in_dataset=args.in_dataset,
#             out_dataset=args.out_dataset, in_server=args.in_server,
#             out_server=args.out_server, work_dir=args.work_dir,
#             parameters=parameters, skip=args.skip, include=args.include,
#             include_bases=include_bases,
#             reprocess=args.reprocess, dataset_depth=args.dataset_depth,
#             modules_env=(args.environment == 'modules'),
#             clean_work_dir=(not args.dont_clean_work_dir))


class GenRefDataCmd():

    desc = ("Generate reference data for a test class, which "
            "future tests will be checked against")

    DEFAULT_TEST_ROOT = op.join(op.dirname(__file__), '..', 'test')

    @classmethod
    def parser(cls):
        parser = ArgumentParser(prog='banana gen-ref', description=cls.desc)
        parser.add_argument('test_class',
                            help=("The path to the test class to test relative"
                                  " to the root test directory (use '.' to "
                                  "separate directories)"))
        parser.add_argument('--specs', default=[], nargs='+',
                            help=("The specs to generate. If not provided all "
                                  "data for all specs will be generated"))
        parser.add_argument('--processor', default=['multi'], nargs='+',
                            metavar='ARG',
                            help=("The type of processor to use plus arguments"
                                  "used to initate it. First arg is the type "
                                  "(one of 'single', 'multi', 'slurm'). "
                                  "Additional arguments depend on type: "
                                  "single [], multi [NUM_PROCS], slurm ["
                                  "ACCOUNT, PARTITION]"))
        parser.add_argument('--environment', type=str, default='static',
                            choices=('modules', 'static'), metavar='TYPE',
                            help="The type of environment to use")
        parser.add_argument('--loggers', nargs='+',
                            default=('nipype.workflow', 'arcana', 'banana'),
                            help="Loggers to set handlers to stdout for")
        parser.add_argument('--reprocess', action='store_true', default=False,
                            help=("Whether to reprocess previously generated "
                                  "derivatives with mismatching provenance"))
        parser.add_argument('--scratch', type=str, default=None,
                            metavar='PATH',
                            help=("The scratch directory to use for the "
                                  "workflow and cache"))
        parser.add_argument('--test_root', default=cls.DEFAULT_TEST_ROOT,
                            help="The location of the root test directory")
        parser.add_argument('--quiet', action='store_true', default=False,
                            help=("Disable logging output"))
        return parser

    @classmethod
    def run(cls, args):

        if not args.quiet:
            set_loggers(args.logger)

        if args.scratch is not None:
            scratch_dir = args.scratch
        else:
            scratch_dir = op.join(op.expanduser('~'), 'banana-scratch')

        # Ensure scratch dir exists
        os.makedirs(scratch_dir, exist_ok=True)

        work_dir = op.join(scratch_dir, 'work')

        proc_args = {'reprocess': args.reprocess}

        if args.processor[0] == 'single':
            processor = SingleProc(work_dir, **proc_args)
        elif args.processor[0] == 'multi':
            if len(args.processor) > 1:
                num_processes = args.processor[1]
            elif len(args.processor) > 2:
                raise BananaUsageError(
                    "Unrecognised arguments passed to '--processor' option "
                    "({}) expected at most 1 additional argument for 'multi' "
                    "type processor (NUM_PROCS)".format(args.processor))
            else:
                num_processes = cpu_count()
            processor = MultiProc(work_dir, num_processes=num_processes,
                                  **proc_args)
        else:
            raise BananaUsageError(
                "Unrecognised processor type provided as first argument to "
                "'--processor' option ({})".format(args.processor[0]))

        if args.environment == 'static':
            environment = StaticEnv()
        else:
            environment = ModulesEnv()

        parts = args.test_class.split('.')

        sys.path.insert(0, op.join(args.test_root, parts[:-2]))
        module = import_module(parts[-2])
        sys.path.pop(0)

        test_cls = getattr(module, parts[-1])

        test_cls().generate_reference_data(
            *args.specs, processor=processor, environment=environment)


class BoxCmd():

    desc = ("Create a containerised pipeline from a given set of inputs to "
            "generate specified derivatives")

    @classmethod
    def parser(cls):
        parser = ArgumentParser(prog='banana box', description=cls.desc)
        parser.add_argument('analysis_class',
                            help=("The path to the analysis class"))
        parser.add_argument('image_name',
                            help=("The name of the Docker image to generate"
                                  "(with org separated by '/')"))
        parser.add_argument('derivatives', nargs='+',
                            help=("The derivatives to generate"))
        parser.add_argument('--input', '-i', action='append', default=[],
                            nargs='+', metavar=('NAME', 'FILE_OR_VAR'),
                            help=("The inputs to provide to the pipeline"))
        parser.add_argument('--switch', '-s', action='append',
                            nargs=2, metavar=('NAME', 'VAR'),
                            help=("Switches provided to the analysis"))
        parser.add_argument('--xnat', action='store_true',
                            help="Insert XNAT CS command JSON as a label")
        parser.add_argument('--maintainer', '-m',
                            help="Maintainer of the pipeline")
        parser.add_argument('--description', '-d', default="Not provided",
                            help="A description of what the pipeline does")
        parser.add_argument('--build', '-b', action='store_true',
                            default=False,
                            help="Build the generated Dockerfile")
        parser.add_argument('--upload', '-u', default=None,
                            help=("Upload the generated dockerfile (requires "
                                  "'-b') to the provided docker index"))
        parser.add_argument('--file_convertors', action='store_true',
                            default=False)
        parser.add_argument('--out_dir', '-o', default=None,
                            help=("The directory to save the Dockerfile to. "
                                  "If not provided then a temp dir will be "
                                  "used instead"))
        return parser

    @classmethod
    def run(cls, args):

        set_loggers([('banana', 'INFO')])

        analysis_class = resolve_class(args.analysis_class)

        inputs = []
        for inpt in args.input:
            if len(inpt) == 1:
                inputs.append(inpt[0])
            elif len(inpt) == 2:
                name, var = inpt
                spec = analysis_class.data_spec(name)
                if spec.is_fileset:
                    inputs.append((name, Fileset.from_path(var)))
                else:
                    inputs.append((name, Field(name, value=var)))
            else:
                raise BananaUsageError(
                    "Inputs should either a single name or a "
                    "name-(filename|value) pair, not ({})".format(inpt))

        switches = {}
        if args.switch:
            for name, val in args.switch:
                spec = analysis_class.param_spec(name)
                switches[name] = spec.dtype(val)

        mock_analysis = analysis_class.mock(inputs=inputs,
                                            switches=switches)

        pipeline_stack = mock_analysis.pipeline_stack(*args.derivatives)

        requirements = defaultdict(set)
        for pipeline in pipeline_stack:
            for node in pipeline.nodes:
                for ver in node.requirements:
                    requirements[ver.requirement].add(ver)

        # Add requirements needed for file conversions
        requirements[mrtrix_req].add(mrtrix_req.v('3.0rc3'))
        requirements[dcm2niix_req].add(dcm2niix_req.v('1.0.20200331'))

        versions = []
        for req, req_versions in requirements.items():
            min_ver, max_ver = req.reconcile(req_versions)
            if req.max_neurodocker_version is not None:
                if req.max_neurodocker_version < min_ver:
                    raise ArcanaRequirementVersionsError(
                        "Minium required version '{}' is greater than max "
                        "neurodocker version ({})".format(
                            min_ver, req.max_neurodocker_version))
                max_ver = req.max_neurodocker_version
            if max_ver is not None:
                version = max_ver
            else:
                # This is where it would be good to know store the latest
                # supported version of each tool so we could use the newest
                # version instead of the min version (i.e. the only one we 
                # know about here
                version = min_ver
            versions.append(version)

        labels = {}

        if args.maintainer:
            labels["maintainer"] = args.maintainer

        parameters = [c.args[0] for c in mock_analysis.parameter.mock_calls]

        if args.xnat:
            if args.upload:
                docker_index = args.upload
            else:
                docker_index = "https://index.docker.io/v1/"
            cmd = XnatCSRepo.command_json(
                args.image_name, analysis_class, inputs, args.derivatives,
                parameters, args.description, docker_index=docker_index)
            print(json.dumps(cmd, indent=2))
            cmd_label = json.dumps(cmd).replace('"', r'\"').replace('$', r'\$')
            labels['org.nrg.commands'] = '[{' + cmd_label + '}]'

        instructions = [
            ["base", "debian:stretch"],
            ["install", ["git", "vim"]]]

        for version in versions:
            props = {'version': str(version)}
            if version.requirement.neurodocker_method:
                props['method'] = version.requirement.neurodocker_method
            instructions.append(
                [version.requirement.neurodocker_name, props])

        if labels:
            instructions.append(["label", labels])

        instructions.append(
            ["miniconda", {
                "create_env": "arcana",
                "conda_install": [
                    "python=3.8",
                    "numpy",
                    "traits"],
                "pip_install": [
                    "git+https://github.com/MonashBI/arcana.git@xnat-cs",
                    "git+https://github.com/MonashBI/banana.git@xnat-cs"]}])

        neurodocker_specs = {
            "pkg_manager": "apt",
            "instructions": instructions}

        dockerfile = neurodocker.Dockerfile(neurodocker_specs).render()

        if args.out_dir is None:
            out_dir = tempfile.mkdtemp()
        else:
            out_dir = args.out_dir

        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, 'Dockerfile')
        with open(out_file, 'w') as f:
            f.write(dockerfile)

        logger.info("Dockerfile generated at %s", out_file)

        if args.build:
            sp.check_call('docker build -t {} .'.format(args.image_name),
                          cwd=out_dir, shell=True)

        if args.upload:
            sp.check_call('docker push {}'.format(args.image_name))


class HelpCmd():

    desc = "Show help for a particular command"

    @classmethod
    def parser(cls):
        parser = ArgumentParser(prog='banana help',
                                description=cls.desc)
        parser.add_argument('command',
                            help="The sub-command to show the help info for")
        return parser

    @classmethod
    def run(cls, args):
        MainCmd.commands[args.command].parser().print_help()


class MenuCmd():

    desc = ("Display the data and parameter specifications for a given analysis "
            "class")

    @classmethod
    def parser(cls):
        parser = ArgumentParser(prog='banana menu',
                                description=cls.desc)
        parser.add_argument('analysis_class',
                            help=("Name of the class to display menu for"))
        parser.add_argument('--full', '-f', action='store_true',
                            help=("Whether to show intermediate derivatives "
                                  "or not"))
        return parser

    @classmethod
    def run(cls, args):
        # Get Analysis class
        analysis_class = resolve_class(args.analysis_class)
        print(analysis_class.static_menu(full=args.full))


class AvailableCmd():

    desc = ("List all available analysis classes within Banana and custom search "
            "paths")

    default_path = 'banana.analysis'

    desc_start = 22

    @classmethod
    def parser(cls):
        parser = ArgumentParser(prog='banana avail',
                                description=cls.desc)
        parser.add_argument('search_paths', nargs='*',
                            help="packages to search for Analysis classes")
        return parser

    @classmethod
    def run(cls, args):
        available = {}

        def find_analysis_classes(pkg_or_module, pkg_or_module_path):
            for cls_name in dir(pkg_or_module):
                if cls_name.startswith('_'):
                    continue
                cls = getattr(pkg_or_module, cls_name)
                try:
                    if (issubclass(cls, (Analysis, MultiAnalysis))
                            and 'desc' in cls.__dict__):
                        try:
                            old_path = available[cls]
                        except KeyError:
                            available[cls] = pkg_or_module_path
                        else:
                            if len(pkg_or_module_path) < len(old_path):
                                available[cls] = pkg_or_module_path
                except TypeError:
                    pass

        search_paths = [cls.default_path] + args.search_paths
        for search_path in search_paths:
            base_module = import_module(search_path)
            for pkg_name in find_packages(op.dirname(base_module.__file__)):
                pkg_path = search_path + '.' + pkg_name
                pkg = import_module(pkg_path)
                find_analysis_classes(pkg, pkg_path)
                for module_info in iter_modules([op.dirname(pkg.__file__)]):
                    module_path = pkg_path + '.' + module_info.name
                    module = import_module(module_path)
                    find_analysis_classes(module, module_path)
        msg = ("\nThe following Analysis classes are available:\n")
        to_print = []
        for avail_cls, module_path in sorted(available.items(),
                                             key=lambda x: x[0].__name__):
            if module_path.startswith(DEFAULT_STUDY_CLASS_PATH):
                module_path = module_path[(len(DEFAULT_STUDY_CLASS_PATH) + 1):]
            full_name = module_path + '.' + avail_cls.__name__
            to_print.append((full_name, avail_cls.desc))

        desc_start = max(len(l[0]) for l in to_print) + DEFAULT_SPACER
        for cls_name, desc in to_print:
            spaces = ' ' * (desc_start - len(cls_name))
            msg += '\n{}{}{}{}'.format(
                ' ' * DEFAULT_INDENT, cls_name, spaces,
                wrap_text(desc, DEFAULT_LINE_LENGTH,
                          desc_start + DEFAULT_INDENT))
        print(msg + '\n')


class MainCmd():

    commands = {
        'avail': AvailableCmd,
        'menu': MenuCmd,
        'derive': DeriveCmd,
        # 'test-gen': TestGenCmd,
        'gen-ref': GenRefDataCmd,
        'box': BoxCmd,
        'help': HelpCmd}

    @classmethod
    def parser(cls):
        usage = "banana <command> [<args>]\n\nAvailable commands:"
        desc_start = max(len(k) for k in cls.commands) + DEFAULT_SPACER
        for name, cmd_cls in cls.commands.items():
            spaces = ' ' * (desc_start - len(name))
            usage += '\n{}{}{}{}'.format(
                ' ' * DEFAULT_INDENT, name, spaces,
                wrap_text(cmd_cls.desc, DEFAULT_LINE_LENGTH,
                          desc_start + DEFAULT_INDENT))
        parser = ArgumentParser(
            description="Base banana command",
            usage=usage)
        parser.add_argument('command', help="The sub-command to run")
        parser.add_argument('--version', '-v', action='version',
                            version='%(prog)s {}'.format(__version__))
        return parser

    @classmethod
    def run(cls, argv=None):
        if argv is None:
            argv = sys.argv[1:]
        parser = cls.parser()
        args = parser.parse_args(argv[:1])
        try:
            cmd_cls = cls.commands[args.command]
        except KeyError:
            print("Unrecognised command '{}'".format(args.command))
            parser.print_help()
            exit(1)
        if args.command == 'help' and len(argv) == 1:
            parser.print_help()
        else:
            cmd_args = cmd_cls.parser().parse_args(argv[1:])
            cmd_cls.run(cmd_args)


if __name__ == '__main__':
    MainCmd.run()
