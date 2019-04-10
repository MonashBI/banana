import os.path
import logging
import tempfile
from unittest import TestCase
from itertools import chain
from arcana.data import FilesetInput, FieldInput
from arcana.repository import BasicRepo
from arcana.exceptions import (
    ArcanaMissingDataException, ArcanaInputMissingMatchError)
from banana.exceptions import BananaTestSetupError
from banana.file_format import niftix_gz_format


logger = logging.getLogger('arcana')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

logging.getLogger("urllib3").setLevel(logging.WARNING)


# class BaseTestCase(ArcanaBaseTestCase):
# 
#     SUBJECT = 'SUBJECT'
#     VISIT = 'VISIT'
#     SERVER = 'https://mbi-xnat.erc.monash.edu.au'
#     XNAT_TEST_PROJECT = 'TEST001'
#     REF_SUFFIX = '_REF'
# 
#     # The path to the test directory, which should sit along side the
#     # the package directory. Note this will not work when Arcana
#     # is installed by a package manager.
#     BASE_TEST_DIR = os.path.abspath(os.path.join(
#         os.path.dirname(banana.__file__), '..', 'test'))


class PipelineTester(TestCase):
    """
    Runs pipelines within a Study class and compares the results
    against reference data from previous runs

    Required class attributes
    -------------------------
    study_class : type(Study)
        The study class run tests against
    ref_repo : Repository
        The repository to draw the reference data from
    parameters : dict[str, str | float | int | datetime]
        The parameters passed to the study when it is initialised
    """

    TEMPDIR_NAME_LEN = 10

    parameters = {}

    def setUp(self):
        if not hasattr(self, 'study_class'):
            raise BananaTestSetupError(
                "{} must have a 'study_class' attribute corresponding to the "
                "Study class to test".format(type(self).__name__))
        if not hasattr(self, 'ref_repo'):
            raise BananaTestSetupError(
                "{} must have a 'ref_repo' attribute corresponding to the "
                "Study class to test".format(type(self).__name__))

        self.name = '{}___{}'.format(
            self.study_class.__name__,
            '__'.join('{}_{}'.format(n, v)
                      for n, v in self.parameters.keys()))
        try:
            base_work_dir = os.environ['BANANA_TEST_WORK_DIR']
        except KeyError:
            self.work_dir = tempfile.mkdtemp()
        else:
            self.work_dir = os.path.join(base_work_dir, self.name)
            os.makedirs(self.work_dir, exist_ok=True)
        self.output_repo = BasicRepo(self.work_dir, depth=0)
        # Create inputs corresponding to each input in the repository
        self.inputs = {}
        for spec in self.study_class.data_specs():
            if spec.is_fileset:
                if not spec.derived and niftix_gz_format in spec.valid_formats:
                    format = niftix_gz_format  # @ReservedAssignment
                else:
                    format = spec.format  # @ReservedAssignment
                selector = FilesetInput(
                    spec.name, spec.name, repository=self.ref_repo,
                    format=format)
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
        self.all_pipelines = set(
            s.pipeline_getter for s in self.study_class.data_specs()
            if hasattr(s, 'pipeline_getter'))

    def pipeline_test(self, pipeline_getter, criteria='exact', dry_run=None):
        if dry_run is None:
            dry_run = self.dry_run
        # A study with all inputs provided to determine which inputs are needed
        # by the pipeline
        input_study = self.study_class(
            pipeline_getter + '_input_checker',
            repository=self.ref_repo,
            processor=self.work_dir,
            inputs=self.inputs.values(),
            parameters=self.parameters)
        try:
            pipeline = getattr(input_study, pipeline_getter)()
        except AttributeError:
            raise BananaTestSetupError(
                "No pipeline named '{}' in {}".format(pipeline_getter,
                                                      self.study_class))
        except ArcanaMissingDataException as e:
            raise BananaTestSetupError(
                "Missing reference data for '{}' required to test '{}' "
                "pipeline in {} with parameters {}".format(
                    e.name, pipeline_getter, self.study_class,
                    self.parameters))
        if not dry_run:
            output_study = self.study_class(
                self.name + '_' + pipeline_getter,
                repository=self.output_repo,
                processor=self.work_dir,
                inputs=[self.inputs[i] for i in pipeline.input_names],
                parameters=self.parameters,
                subject_ids=input_study.subject_ids,
                visit_ids=input_study.visit_ids,
                enforce_inputs=False,
                fill_tree=True)
            pipeline = output_study.pipeline(pipeline_getter)
            for spec_name in pipeline.output_names:
                for ref, test in zip(input_study.data(spec_name),
                                     output_study.data(spec_name)):
                    if ref.is_fileset:
                        self.assertEqual(
                            ref.checksums, test.checksums,
                            "checksums don't match reference for {} generated "
                            "by {} in {}".format(spec_name, pipeline_getter,
                                                 self.study_class))
                    else:
                        self.assertEqual(
                            ref.checksums, test.checksums,
                            "value for {} ({}) generated by {} in {} doesn't "
                            "match reference ({})".format(
                                spec_name, pipeline_getter, self.study_class))

    def test_all(self, skip=()):
        return list(chain(
            self.pipeline_test(p) for p in self.all_pipelines
            if p not in skip))

    @classmethod
    def test_data_description(cls, study_class, parameters):
        return (
            'Test data for {} Banana class with the following non-default '
            'parameters:\n{}'.format(
                study_class.__name__,
                '\n'.join('{}={}'.format(k, v)
                          for k, v in parameters.items())))

