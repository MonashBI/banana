import sys
import os.path as op
import os
from argparse import ArgumentParser
from importlib import import_module
from banana.utils.testing import PipelineTester
from banana.exceptions import BananaUsageError
from multiprocessing import cpu_count
from arcana.utils import parse_value
from banana import (
    InputFilesets, MultiProc, SingleProc, SlurmProc, StaticEnv, ModulesEnv,
    BasicRepo, BidsRepo, XnatRepo)
import logging

logger = logging.getLogger('banana')


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


def resolve_class(class_str, prefixes=('banana.', 'banana.study.')):
    parts = class_str.split('.')
    module_name = '.'.join(parts[:-1])
    class_name = parts[-1]
    cls = None
    for prefix in [''] + list(prefixes):
        mod_name = prefix + module_name
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

    desc = "Generate derivatives from a Banana Study class"

    @classmethod
    def parser(cls):
        parser = ArgumentParser(prog='banana derive',
                                description=cls.desc)
        parser.add_argument('repository_path',
                            help=("Either the path to the repository if of "
                                  "'bids' or 'basic' types, or the name of the"
                                  " project ID for 'xnat' type"))
        parser.add_argument('study_class',
                            help=("Name of the class to instantiate"))
        parser.add_argument('study_name',
                            help=("The name of the study to put the analysis "
                                  "under (e.g. parenthood)"))
        parser.add_argument('derivatives', nargs='+',
                            help=("The names of the derivatives to generate"))
        parser.add_argument('--repository', nargs='+', default=['bids'],
                            metavar='ARG',
                            help=("Specify the repository type and any options"
                                  " to be passed to it. First argument "))
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
        parser.add_argument('--input', '-i', nargs=2, action='append',
                            default=[], metavar=('SPEC', 'PATTERN'),
                            help=("The inputs to include in the study init. "
                                  "If not provided then all are used"))
        parser.add_argument('--parameter', '-p', nargs=2, action='append',
                            metavar=('NAME', 'VALUE'), default=[],
                            help="Parameters to pass to the study")
        parser.add_argument('--subject_ids', nargs='+', default=None,
                            metavar='ID',
                            help=("The subject IDs to include in the analysis."
                                  " If a single value with a '/' in it is "
                                  "provided then it is interpreted as a text "
                                  "file containing a list of IDs"))
        parser.add_argument('--visit_ids', nargs='+', default=None,
                            metavar='ID',
                            help=("The visit IDs to include in the analysis"
                                  "If a single value with a '/' in it is "
                                  "provided then it is interpreted as a text "
                                  "file containing a list of IDs"))
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
        return parser

    @classmethod
    def run(cls, args):

        set_loggers(args.logger)

        study_class = resolve_class(args.study_class)

        if args.scratch is not None:
            scratch_dir = args.scratch
        else:
            scratch_dir = op.join(op.expanduser('~'), 'banana-scratch')

        # Ensure scratch dir exists
        os.makedirs(scratch_dir, exist_ok=True)

        work_dir = op.join(scratch_dir, 'work')

        if args.repository is None:
            if args.input:
                repo_type = 'basic'
            else:
                repo_type = 'bids'
        else:
            repo_type = args.repository[0]

        if repo_type == 'bids':
            repository = BidsRepo(args.repository_path)
        elif repo_type == 'basic':
            if len(args.repository) != 2:
                raise BananaUsageError(
                    "Unrecognised arguments passed to '--repository' option "
                    "({}) exactly 1 additional argument is required for "
                    "'basic' type repository (DEPTH)".format(args.respository))
            repository = BasicRepo(args.repository_path)
        elif repo_type == 'xnat':
            nargs = len(args.repository)
            if nargs < 2:
                raise BananaUsageError(
                    "Not enough arguments passed to '--repository' option "
                    "({}), at least 1 additional argument is required for "
                    "'xnat' type repository (SERVER)"
                    .format(args.respository))
            elif nargs > 4:
                raise BananaUsageError(
                    "Unrecognised arguments passed to '--repository' option "
                    "({}), at most 3 additional argument are accepted for "
                    "'xnat' type repository (SERVER, USER, PASSWORD)"
                    .format(args.respository))
            repository = XnatRepo(
                project_id=args.repository_path,
                server=args.repository[1],
                user=(args.repository[2] if nargs > 2 else None),
                password=(args.repository[3] if nargs > 3 else None),
                cache_dir=op.join(scratch_dir, 'cache'))
        else:
            raise BananaUsageError(
                "Unrecognised repository type provided as first argument to "
                "'--repository' option ({})".format(args.repository[0]))

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

        # Load subject_ids from file if single value is provided with
        # a '/' in the string
        if (args.subject_ids is not None and len(args.subject_ids) and
                '/' in args.subject_ids[0]):
            with open(args.subject_ids[0]) as f:
                subject_ids = f.read().split()
        else:
            subject_ids = args.subject_ids

        # Load visit_ids from file if single value is provided with
        # a '/' in the string
        if (args.visit_ids is not None and len(args.visit_ids) and
                '/' in args.visit_ids[0]):
            with open(args.visit_ids[0]) as f:
                visit_ids = f.read().split()
        else:
            visit_ids = args.visit_ids

        parameters = {}
        for name, value in args.parameter:
            parameters[name] = parse_value(value)

        study = study_class(
            name=args.study_name,
            repository=repository,
            processor=processor,
            environment=environment,
            inputs=dict(args.input),
            parameters=parameters,
            subject_ids=subject_ids,
            visit_ids=visit_ids,
            enforce_inputs=args.enforce_inputs)

        for spec_name in args.cache:
            spec = study.bound_spec(spec_name)
            if not isinstance(spec, InputFilesets):
                raise BananaUsageError(
                    "Cannot cache non-input fileset '{}'".format(spec_name))
            spec.cache()

        # Generate data
        study.data(args.derivatives)

        logger.info("Generated derivatives for '{}'".format(args.derivatives))


class TestDataCmd():

    desc = "Generate derivatives from a study"

    @classmethod
    def parser(cls):
        parser = ArgumentParser(
            prog='banana test_data',
            description=("Generates reference data for a pipeline tester "
                         "unittests given a study class and set of "
                         "parameters"))
        parser.add_argument('study_class',
                            help=("The path to the study class to test, e.g. "
                                  "banana.study.MriStudy"))
        parser.add_argument('in_repo', help=("The path to repository that "
                                             "houses the input data"))
        parser.add_argument('out_repo',
                            help=("If the 'xnat_server' argument is provided "
                                  "then out is interpreted as the project ID "
                                  "to use the XNAT server (the project must "
                                  "exist already). Otherwise it is interpreted"
                                  " as the path to a basic repository"))
        parser.add_argument('--in_server', default=None,
                            help="The server to download the input data from")
        parser.add_argument('--out_server', default=None,
                            help="The server to upload the reference data to")
        parser.add_argument('--work_dir', default=None,
                            help="The work directory")
        parser.add_argument('--parameter', '-p', metavar=('NAME', 'VALUE'),
                            nargs=2, action='append', default=[],
                            help=("Parameters to set when initialising the "
                                  "study"))
        parser.add_argument('--skip', '-s', nargs='+', default=[],
                            help=("Spec names to skip in the generation "
                                  "process"))
        parser.add_argument('--skip_base', action='append', default=[],
                            help=("Base classes of which to skip data specs "
                                  "from"))
        parser.add_argument('--reprocess', action='store_true', default=False,
                            help=("Whether to reprocess previously generated "
                                  "datasets in the output repository"))
        parser.add_argument('--repo_depth', type=int, default=0,
                            help="The depth of the input repository")
        parser.add_argument('--modules_env', action='store_true',
                            default=False,
                            help="Whether to use a Modules Envionment or not")
        parser.add_argument('--dont_clean_work_dir', action='store_true',
                            default=False,
                            help=("Whether to clean the Nipype work dir "
                                  "between runs"))
        parser.add_argument('--loggers', nargs='+',
                            default=('nipype.workflow', 'arcana', 'banana'),
                            help="Loggers to set handlers to stdout for")
        return parser

    @classmethod
    def run(cls, args):

        # Get Study class
        study_class = resolve_class(args.study_class)

        include_bases = [resolve_class(c) for c in args.skip_base]

        # Convert parameters to dictionary
        parameters_dct = {}
        for name, value in args.parameter:
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass
            parameters_dct[name] = value
        parameters = parameters_dct

        PipelineTester.generate_test_data(
            study_class=study_class, in_repo=args.in_repo,
            out_repo=args.out_repo, in_server=args.in_server,
            out_server=args.out_server, work_dir=args.work_dir,
            parameters=parameters, skip=args.skip, include_bases=include_bases,
            reprocess=args.reprocess, repo_depth=args.repo_depth,
            modules_env=args.modules_env,
            clean_work_dir=(not args.dont_clean_work_dir))


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

    desc = ("Display the data and parameter specifications for a given study "
            "class")

    @classmethod
    def parser(cls):
        parser = ArgumentParser(prog='banana menu',
                                description=cls.desc)
        parser.add_argument('study_class',
                            help=("Name of the class to display menu for"))
        return parser

    @classmethod
    def run(cls, args):
        # Get Study class
        study_class = resolve_class(args.study_class)
        print(study_class.static_menu())


class MainCmd():

    commands = {
        'derive': DeriveCmd,
        'menu': MenuCmd,
        'test_data': TestDataCmd,
        'help': HelpCmd}

    @classmethod
    def parser(cls):
        usage = "banana <command> [<args>]\n\nAvailable commands:"
        for name, cmd_cls in cls.commands.items():
            usage += '\n\t{}\t\t{}'.format(name, cmd_cls.desc)
        parser = ArgumentParser(
            description="Base banana command",
            usage=usage)
        parser.add_argument('command', help="The sub-command to run")
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
        cmd_args = cmd_cls.parser().parse_args(argv[1:])
        cmd_cls.run(cmd_args)


if __name__ == '__main__':
    MainCmd.run()
