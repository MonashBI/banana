import os
import sys
import tempfile
from itertools import chain
from traceback import format_exception
from arcana.data import FilesetSelector, FieldSelector
from arcana.repository import DirectoryRepository
from arcana.exceptions import ArcanaMissingDataException
from banana.exceptions import BananaTestSetupError


class SystemTester(object):
    """
    Runs pipelines within a Study class and compares the results
    against reference data from previous runs
    """

    def __init__(self, study_class, ref_repo, work_dir=None, parameters=None,
                 dry_run=False, missing_ok=False):
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
        self.output_repo = DirectoryRepository(work_dir, depth=0)
        # Create inputs corresponding to each input in the repository
        self.inputs = {}
        for spec in study_class.data_specs():
            if spec.is_fileset:
                selector = FilesetSelector(
                    spec.name, spec.name, format=spec.format,
                    repository=self.ref_repo)
            else:
                selector = FieldSelector(
                    spec.name, spec.name, dtype=spec.dtype,
                    repository=self.ref_repo)
            # Check whether a corresponding data exists in the reference repo
            try:
                selector.bind(ref_repo)
            except ArcanaMissingDataException:
                if not missing_ok:
                    raise
            else:
                self.inputs[spec.name] = selector
        self.all_pipelines = set(
            s.pipeline_name for s in study_class.data_specs())
        self.dry_run = dry_run

    def test(self, pipeline_name, criteria='exact', dry_run=None):
        if dry_run is None:
            dry_run = self.dry_run
        # A study with all inputs provided to determine which inputs are needed
        # by the pipeline
        input_study = self.study_class(
            pipeline_name + '_input_checker',
            repo=self.ref_repo,
            processor=self.work_dir,
            inputs=self.inputs.values(),
            parameters=self.parameters)
        try:
            pipeline = getattr(input_study, pipeline_name)()
        except AttributeError:
            raise BananaTestSetupError(
                "No pipeline named '{}' in {}".format(pipeline_name,
                                                      self.study_class))
        except ArcanaMissingDataException as e:
            raise BananaTestSetupError(
                "Missing reference data for '{}' required to test '{}' "
                "pipeline in {} with parameters {}".format(
                    e.name, pipeline_name, self.study_class, self.parameters))
        if not dry_run:
            output_study = self.study_class(
                self.name + '_' + pipeline_name,
                repo=self.output_repo,
                processor=self.work_dir,
                inputs=[self.inputs[i] for i in pipeline.input_names],
                parameters=self.parameters)
            pipeline = getattr(output_study, pipeline_name)()
            try:
                pipeline.run()
            except Exception:
                results = [TestError(pipeline_name,
                                     format_exception(*sys.exc_info()))]
            else:
                results = []
            for spec_name in pipeline.output_names:
                if input_study.data(spec_name) != output_study.data(spec_name):
                    results.append(
                        TestFailure(pipeline_name, spec_name, None, None))
                else:
                    results.append(TestSuccess(pipeline_name, spec_name))
            return results

    def test_all(self, skip=()):
        return list(chain(
            self.test(p) for p in self.all_pipelines if p not in skip))


class TestSuccess(object):

    type = 'success'

    def __init__(self, pipeline_name, spec_name, study_class):
        self.pipeline_name = pipeline_name
        self.spec_name = spec_name
        self.study_class = study_class


class TestFailure(TestSuccess):

    type = 'failure'

    def __init__(self, pipeline_name, spec_name, study_class,
                 derived_value, ref_value):
        super().__init__(pipeline_name, spec_name, study_class)
        self.derived_value = derived_value
        self.ref_value = ref_value

    def __repr__(self):
        return ("FAILURE! {} did not match reference produced by {}.{}"
                .format(self.spec_name, self.study_class.__name__,
                        self.pipeline_name))


class TestError(object):

    type = 'error'

    def __init__(self, pipeline_name, study_class, traceback):
        self.pipeline_name = pipeline_name
        self.study_class = study_class
        self.traceback = traceback

    def __repr__(self):
        return ("ERROR! running {}.{}:\n{}"
                .format(self.study_class.__name__, self.pipeline_name,
                        self.traceback))
