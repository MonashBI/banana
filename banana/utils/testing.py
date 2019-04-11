import os
import os.path as op
import logging
import tempfile
from itertools import chain
from unittest import TestCase
from arcana import FilesetInput, FieldInput, BasicRepo, SingleProc
from arcana.exceptions import ArcanaInputMissingMatchError
from banana.exceptions import BananaTestSetupError


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
                selector.match(self.ref_repo.cached_tree())
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
                    self.assertEqual(
                        test.checksums, ref.checksums,
                        "checksums don't match reference for {} generated "
                        "by {} in {}".format(spec_name, pipeline_getter,
                                             self.study_class))
                else:
                    self.assertEqual(
                        test.value, ref.value,
                        "value for {} ({}) generated by {} in {} doesn't "
                        "match reference ({})".format(
                            spec_name, test.value, pipeline_getter,
                            self.study_class, ref.value))
