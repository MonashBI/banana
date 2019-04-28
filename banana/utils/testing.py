import os
import os.path as op
import logging
import tempfile
from copy import copy
import json
from pprint import pformat
from itertools import chain
from unittest import TestCase
from argparse import ArgumentParser
from importlib import import_module
from arcana.exceptions import ArcanaNameError
from arcana import (FilesetInput, FieldInput, BasicRepo, XnatRepo, SingleProc,
                    Field, Fileset)
from arcana.exceptions import ArcanaInputMissingMatchError
from banana.exceptions import BananaTestSetupError, BananaUsageError
import banana.file_format  # @UnusedImport


logger = logging.getLogger('arcana')
logger.setLevel(logging.WARNING)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

logging.getLogger("urllib3").setLevel(logging.WARNING)

try:
    TEST_DIR = os.environ['BANANA_TEST_DIR']
except KeyError:
    TEST_DIR = tempfile.mkdtemp()

TEST_CACHE_DIR = op.join(TEST_DIR, 'cache')


class PipelineTester(TestCase):
    """
    Runs pipelines within a Study class and compares the results
    against reference data from previous runs

    Class attributes
    ----------------
    name : str
        A unique name (among other unittests to be run in same batch) for the
        test class
    study_class : type(Study)
        The study class run tests against
    ref_repo : Repository
        The repository to draw the reference data from
    parameters : dict[str, str | float | int | datetime]
        The parameters passed to the study when it is initialised
    warn_missing_tests : bool
        Whether to raise a warning if there is a pipeline in the study class
        that does not have a test in tester class
    """

    parameters = {}
    warn_missing_tests = True

    def setUp(self):
        if not hasattr(self, 'study_class'):
            raise BananaTestSetupError(
                "{} must have a 'study_class' attribute corresponding to the "
                "Study class to test".format(type(self).__name__))
        if not hasattr(self, 'ref_repo'):
            raise BananaTestSetupError(
                "{} must have a 'ref_repo' attribute corresponding to the "
                "Study class to test".format(type(self).__name__))
        base_test_dir = op.join(TEST_DIR, self.name)
        self.work_dir = op.join(base_test_dir, 'work')
        repo_root = op.join(base_test_dir, 'repo')
        os.makedirs(self.work_dir, exist_ok=True)
        os.makedirs(repo_root, exist_ok=True)
        # Create repository to hold outputs
        self.output_repo = BasicRepo(repo_root, depth=2)
        # Create inputs for reference study
        self.inputs = {}
        for spec in self.study_class.data_specs():
            # Create an input for each entry in the class specificiation
            if spec.is_fileset:
                selector = FilesetInput(
                    spec.name, spec.name, repository=self.ref_repo)
            else:
                selector = FieldInput(
                    spec.name, spec.name, dtype=spec.dtype,
                    repository=self.ref_repo)
            # Check whether a corresponding data exists in the reference repo
            try:
                selector.match(self.ref_repo.cached_tree(), spec)
            except ArcanaInputMissingMatchError:
                continue
            self.inputs[spec.name] = selector
        # Create the reference study
        self.ref_study = self.study_class(
            self.name,
            repository=self.ref_repo,
            processor=self.work_dir,
            inputs=self.inputs.values(),
            parameters=self.parameters)
        # Get set of all pipelines to test
        if self.warn_missing_tests:
            pipelines_to_test = set(
                s.pipeline_getter for s in self.study_class.data_specs()
                if s.derived)
            test_names = [m for m in dir(self) if m.startswith('test_')]
            for pipeline_getter in pipelines_to_test:
                expected_test_name = 'test_' + pipeline_getter
                if expected_test_name not in test_names:
                    logger.warning("Did not find test for '{}' pipeline in {}"
                                   "tester. Add to 'suppress_warning' class "
                                   "attr to suppress this message".format(
                                       pipeline_getter, self.name))

    def run_pipeline_test(self, pipeline_getter, add_inputs=(),
                          test_criteria=None):
        """
        Runs a pipeline and tests its outputs against the reference data

        Parameters
        ----------
        pipeline_getter : str
            The name of the pipeline to test
        add_inputs : list[str]
            Inputs that are required in the output study for the pipeline to
            run, in addition to the direct inputs of the pipeline, i.e. ones
            that are tested for with the 'provided' method in the pipeline
            construction
        test_criteria : dct[str, *] | None
            A dictionary containing the criteria by which to determine equality
            for the derived filesets. The keys are spec-names and the values
            are specific to the format of the fileset. If a spec-name is not
            in the dictionary or None is provided then the default
            criteria are used for each fileset test.
        """
        if test_criteria is None:
            test_criteria = {}
        # A study with all inputs provided to determine which inputs are needed
        # by the pipeline
        ref_pipeline = self.ref_study.pipeline(pipeline_getter)
        inputs = []
        for spec_name in chain(ref_pipeline.input_names, add_inputs):
            try:
                inputs.append(self.inputs[spec_name])
            except KeyError:
                pass  # Inputs with a default value
        # Set up output study
        output_study = self.study_class(
            pipeline_getter,
            repository=self.output_repo,
            processor=SingleProc(self.work_dir, reprocess='force'),
            inputs=inputs,
            parameters=self.parameters,
            subject_ids=self.ref_study.subject_ids,
            visit_ids=self.ref_study.visit_ids,
            enforce_inputs=False,
            fill_tree=True)
        for spec_name in ref_pipeline.output_names:
            for ref, test in zip(self.ref_study.data(spec_name),
                                 output_study.data(spec_name)):
                if ref.is_fileset:
                    try:
                        self.assertTrue(
                            test.contents_equal(
                                ref, **test_criteria.get(spec_name, {})),
                            "'{}' fileset generated by {} in {} doesn't match "
                            "reference".format(spec_name, pipeline_getter,
                                               self.study_class))
                    except Exception:
                        if hasattr(test, 'headers_diff'):
                            header_diff = test.headers_diff(ref)
                            if header_diff:
                                print("Headers don't match on {}"
                                      .format(header_diff))
                                print("Test header:\n{}".format(
                                    pformat(test.get_header())))
                                print("Reference header:\n{}".format(
                                    pformat(ref.get_header())))
                            else:
                                print("Image RMS diff: {}"
                                      .format(test.rms_diff(ref)))
                                test.contents_equal(ref)
                        raise
                else:
                    self.assertEqual(
                        test.value, ref.value,
                        "value for {} ({}) generated by {} in {} doesn't "
                        "match reference ({})".format(
                            spec_name, test.value, pipeline_getter,
                            self.study_class, ref.value))

    @classmethod
    def generate_test_data(cls, study_class, in_repo, out_repo,
                           in_server=None, out_server=None, work_dir=None,
                           parameters=(), include=None, skip=(),
                           reprocess=False, repo_depth=0):
        """
        Generates reference data for a pipeline tester unittests given a study
        class and set of parameters

        Parameters
        ----------
        study_class : type(Study)
            The path to the study class to test, e.g. banana.study.MriStudy
        in_repo : str
            The path to repository that houses the input data
        out_repo : str
            If the 'xnat_server' argument is provided then out
            is interpreted as the project ID to use the XNAT
            server (the project must exist already). Otherwise
            it is interpreted as the path to a basic repository
        in_server : str | None
            The server to download the input data from
        out_server : str | None
            The server to upload the reference data to
        work_dir : str
            The work directory
        parameters : dict[str, *]
            Parameter to set when initialising the study
        include : list[str] | None
            Spec names to include in the output repository. If None all names
            except those listed in 'skip' are included
        skip : list[str]
            Spec names to skip in the generation process. Only valid if
            'include' is None
        reprocess : bool
            Whether to reprocess the generated datasets
        repo_depth : int
            The depth of the input repository
        """
        if work_dir is None:
            work_dir = tempfile.mkdtemp()
        else:
            work_dir = work_dir

        parts = study_class.split('.')
        module_name = '.'.join(parts[:-1])
        class_name = parts[-1]

        # Convert parameters to dictionary
        parameters_dct = {}
        for name, value in parameters:
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass
            parameters_dct[name] = value
        parameters = parameters_dct

        module = import_module(module_name)
        study_class = getattr(module, class_name)

        if class_name.endswith('Study'):
            study_name = class_name[:-len('Study')]
        else:
            study_name = class_name

        # Get output repository to write the data to
        if in_server is not None:
            in_repo = XnatRepo(project_id=in_repo, server=in_server,
                               cache_dir=op.join(work_dir, 'xnat-cache'))
        else:
            in_repo = BasicRepo(in_repo, depth=repo_depth)

        temp_repo = BasicRepo(op.join(work_dir, 'temp-repo'), depth=repo_depth)

        in_paths = []
        inputs = []
        for fname in os.listdir(in_repo):
            if fname == study_name:
                continue
            path = op.join(in_repo.root_dir, fname)
            in_paths.append(path)
            if fname == BasicRepo.FIELDS_FNAME:
                with open(path) as f:
                    field_data = json.load(f)
                for name, value in field_data.items():
                    field = Field(name, value)
                    selector = FieldInput(field.name, field.name, field.dtype,
                                          repository=in_repo)
                    inputs.append(selector)
            else:
                fileset = Fileset.from_path(path)
                selector = FilesetInput(fileset.basename, fileset.basename,
                                        fileset.format, repository=in_repo)
                try:
                    spec = study_class.data_spec(selector)
                except ArcanaNameError:
                    print("Skipping '{}' as it doesn't match a spec in {}"
                          .format(fname, study_class))
                    continue
                if not (spec.derived and
                        selector.format in spec.valid_formats):
                    print("Skipping '{}' as it doesn't have a valid format "
                          "for {} in {} ({})".format(
                              fname, spec.name, study_class,
                              ', '.join(f.name for f in spec.valid_formats)))
                    continue
                inputs.append(selector)

        study = study_class(
            study_name,
            repository=temp_repo,
            processor=SingleProc(work_dir, reprocess=reprocess),
            inputs=inputs,
            parameters=parameters,
            fill_tree=True)

        if include is None:
            include = [n for n in study.data_spec_names() if n not in skip]
        elif skip:
            raise BananaUsageError(
                "Cannot provide both 'include' and 'skip' options to "
                "PipelineTester.generate_test_data")

        # Generate all derived data
        for spec_name in include:
            study.data(spec_name)

        # Get output repository to write the data to
        if out_server is not None:
            out_repo = XnatRepo(project_id=out_repo, server=in_server,
                                cache_dir=op.join(work_dir, 'xnat-cache'))
        else:
            out_repo = BasicRepo(out_repo, depth=repo_depth)

        # Upload data to repository
        for spec in study.data_specs():
            if spec.name not in skip:
                for item in study.data(spec.name):
                    if not item.exists:
                        continue
                    if item.is_fileset:
                        item_cpy = Fileset(
                            name=item.name, format=item.format,
                            frequency=item.frequency, path=item.path,
                            aux_files=copy(item.aux_files),
                            subject_id=item.subject_id, visit_id=item.visit_id,
                            repository=out_repo, exists=True)
                    else:
                        item_cpy = Field(
                            name=item.name, value=item.value, dtype=item.dtype,
                            frequency=item.frequency, array=item.array,
                            subject_id=item.subject_id, visit_id=item.visit_id,
                            repository=out_repo, exists=True)
                    item_cpy.put()


def gen_test_data_entry_point():
    parser = ArgumentParser(
        description=("Generates reference data for a pipeline tester "
                     "unittests given a study class and set of parameters"))
    parser.add_argument('study_class',
                        help=("The path to the study class to test, e.g. "
                              "banana.study.MriStudy"))
    parser.add_argument('in_repo', help=("The path to repository that "
                                         "houses the input data"))
    parser.add_argument('out_repo',
                        help=("If the 'xnat_server' argument is provided then "
                              "out is interpreted as the project ID to use "
                              "the XNAT server (the project must exist "
                              "already). Otherwise it is interpreted as the "
                              "path to a basic repository"))
    parser.add_argument('--xnat_server', default=None,
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
    parser.add_argument('--reprocess', action='store_true', default=False,
                        help=("Whether to reprocess previously generated "
                              "datasets in the output repository"))
    parser.add_argument('--repo_depth', type=int, default=0,
                        help="The depth of the input repository")
    args = parser.parse_args()

    PipelineTester.generate_test_data(
        study_class=args.study_class, in_repo=args.in_repo,
        out_repo=args.out_repo, xnat_server=args.xnat_server,
        work_dir=args.work_dir, parameters=args.parameter, skip=args.skip,
        reprocess=args.reprocess, repo_depth=args.repo_depth)
