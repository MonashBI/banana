import os
import sys
import tempfile
from itertools import chain
from traceback import format_exception
from arcana.data import FilesetSelector, FieldSelector
from arcana.repository import DirectoryRepository
from arcana.exceptions import (
    ArcanaMissingDataException, ArcanaSelectorMissingMatchError)
from banana.exceptions import BananaTestSetupError
from banana.file_format import niftix_gz_format


class SystemTester(object):
    """
    Runs pipelines within a Study class and compares the results
    against reference data from previous runs
    """

    def __init__(self, study_class, ref_repo, work_dir=None, parameters=None,
                 dry_run=False):
        if parameters is None:
            parameters = {}
        self.parameters = parameters
        self.name = '{}___{}'.format(study_class.__name__,
                                     '__'.join('{}_{}'.format(n, v)
                                               for n, v in parameters.keys()))
        self.study_class = study_class
        if work_dir is None:
            self.work_dir = tempfile.mkdtemp()
        else:
            self.work_dir = work_dir
            os.makedirs(self.work_dir, exist_ok=True)
        self.ref_repo = ref_repo
        self.output_repo = DirectoryRepository(self.work_dir, depth=0)
        # Create inputs corresponding to each input in the repository
        self.inputs = {}
        for spec in study_class.data_specs():
            if spec.is_fileset:
                if not spec.derived and niftix_gz_format in spec.valid_formats:
                    format = niftix_gz_format  # @ReservedAssignment
                else:
                    format = spec.format  # @ReservedAssignment
                selector = FilesetSelector(
                    spec.name, spec.name, repository=self.ref_repo,
                    format=format)
            else:
                selector = FieldSelector(
                    spec.name, spec.name, dtype=spec.dtype,
                    repository=self.ref_repo)
            # Check whether a corresponding data exists in the reference repo
            try:
                selector.match(ref_repo.cached_tree())
            except ArcanaSelectorMissingMatchError:
                continue
            self.inputs[spec.name] = selector
        self.all_pipelines = set(
            s.pipeline_getter for s in study_class.data_specs()
            if hasattr(s, 'pipeline_getter'))
        self.dry_run = dry_run

    def test(self, pipeline_getter, criteria='exact', dry_run=None):
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
            try:
                output_study.processor.run(pipeline)
            except Exception:
                results = [TestError(pipeline_getter, self.study_class,
                                     format_exception(*sys.exc_info()))]
            else:
                results = []
                for spec_name in pipeline.output_names:
                    ref_data = next(iter(input_study.data(spec_name)))
                    test_data = next(iter(output_study.data(spec_name)))
                    fail = False
                    if ref_data.is_fileset:
                        if ref_data.checksums != test_data.checksums:
                            fail = True
                            msg = "checksums don't match"
                    else:
                        if ref_data.value != test_data.value:
                            fail = True
                            msg = (
                                "generated value ({}) did not match reference "
                                "({})".format(test_data.value, ref_data.value))
                    if fail:
                        results.append(
                            TestFailure(pipeline_getter, spec_name,
                                        self.study_class, msg))
                    else:
                        results.append(TestSuccess(pipeline_getter, spec_name,
                                                   self.study_class))
            return results

    def test_all(self, skip=()):
        return list(chain(
            self.test(p) for p in self.all_pipelines if p not in skip))

    @classmethod
    def print_results(cls, results):
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
            print("{} tests succeeded".format(len(successes)))
        else:
            print("{} errors and {} failures".format(len(errors),
                                                     len(failures)))


class TestSuccess(object):

    type = 'success'

    def __init__(self, pipeline_name, spec_name, study_class):
        self.pipeline_name = pipeline_name
        self.spec_name = spec_name
        self.study_class = study_class


class TestFailure(TestSuccess):

    type = 'failure'

    def __init__(self, pipeline_name, spec_name, study_class, msg):
        super().__init__(pipeline_name, spec_name, study_class)
        self.msg = msg

    def __repr__(self):
        return ("FAILURE! {} did not match reference produced by {}.{}: {}"
                .format(self.spec_name, self.study_class.__name__,
                        self.pipeline_name, self.msg))


class TestError(object):

    type = 'error'

    def __init__(self, pipeline_name, study_class, traceback):
        self.pipeline_name = pipeline_name
        self.study_class = study_class
        self.traceback = traceback

    def __repr__(self):
        return ("ERROR! running {}.{}:\n{}"
                .format(self.study_class.__name__, self.pipeline_name,
                        ','.join(self.traceback)))
