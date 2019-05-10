import os
import os.path as op
import logging
import tempfile
import shutil
from copy import copy
from pprint import pformat
from itertools import chain
from unittest import TestCase
from argparse import ArgumentParser
from importlib import import_module
from arcana.exceptions import ArcanaNameError
from arcana import (InputFileset, InputField, BasicRepo, XnatRepo, SingleProc,
                    Field, Fileset)
from arcana.exceptions import (
    ArcanaInputMissingMatchError, ArcanaMissingDataException)
from banana.exceptions import BananaTestSetupError, BananaUsageError
import banana.file_format  # @UnusedImport


logger = logging.getLogger('arcana')
logger.setLevel(logging.WARNING)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(levelname)s - %(message)s")
handler.setFormatter(formatter)

# logger = logging.getLogger('nipype.workflow')
# logger.setLevel(logging.DEBUG)
# handler = logging.StreamHandler()
# formatter = logging.Formatter("%(levelname)s - %(message)s")
# handler.setFormatter(formatter)

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
                inpt = InputFileset(
                    spec.name, spec.name, repository=self.ref_repo)
            else:
                inpt = InputField(
                    spec.name, spec.name, dtype=spec.dtype,
                    repository=self.ref_repo)
            # Check whether a corresponding data exists in the reference repo
            try:
                inpt.match(self.ref_repo.cached_tree(), spec)
            except ArcanaInputMissingMatchError:
                continue
            self.inputs[spec.name] = inpt
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
                          test_criteria=None, pipeline_args=None):
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
        if pipeline_args is None:
            pipeline_args = {}
        # A study with all inputs provided to determine which inputs are needed
        # by the pipeline
        ref_pipeline = self.ref_study.pipeline(pipeline_getter,
                                               pipeline_args=pipeline_args)
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
                           parameters=(), include=None, skip=(), skip_bases=(),
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
        skip_bases : list[type(Study)]
            List of base classes in which all entries in their data
            specification that is not explicitly in the test data class is
            added to the list of specs to skip
        reprocess : bool
            Whether to reprocess the generated datasets
        repo_depth : int
            The depth of the input repository
        """
        if work_dir is None:
            work_dir = tempfile.mkdtemp()
        else:
            work_dir = work_dir

        if study_class.__name__.endswith('Study'):
            study_name = study_class.__name__[:-len('Study')]
        else:
            study_name = study_class.__name__

        # Get output repository to write the data to
        if in_server is not None:
            in_repo = XnatRepo(project_id=in_repo, server=in_server,
                               cache_dir=op.join(work_dir, 'xnat-cache'))
        else:
            in_repo = BasicRepo(in_repo, depth=repo_depth)

        temp_repo_root = op.join(work_dir, 'temp-repo')
        if os.path.exists(temp_repo_root) and reprocess:
            shutil.rmtree(temp_repo_root)
        os.makedirs(temp_repo_root, exist_ok=True)

        temp_repo = BasicRepo(temp_repo_root, depth=repo_depth)

        inputs = None
        for session in in_repo.tree().sessions:
            session_inputs = []
            for item in chain(session.filesets, session.fields):
                if isinstance(item, Fileset):
                    inpt = InputFileset(item.basename, item.basename,
                                        item.format, repository=in_repo)
                else:
                    inpt = InputField(item.name, item.name, item.dtype,
                                      repository=in_repo)
                try:
                    spec = study_class.data_spec(inpt)
                except ArcanaNameError:
                    print("Skipping {} as it doesn't match a spec in {}"
                          .format(item, study_class))
                else:
                    session_inputs.append(inpt)
            session_inputs = sorted(session_inputs)
            if inputs is not None and session_inputs != inputs:
                raise BananaUsageError(
                    "Inconsistent inputs ({} and {}) found in sessions of {}"
                    .format(inputs, session_inputs, in_repo))
            else:
                inputs = session_inputs

        study = study_class(
            study_name,
            repository=temp_repo,
            processor=SingleProc(
                work_dir, reprocess=reprocess, prov_ignore=(
                    SingleProc.DEFAULT_PROV_IGNORE +
                    ['.*/pkg_version',
                     'workflow/nodes/.*/requirements/.*/version'])),
            inputs=inputs,
            parameters=parameters,
            subject_ids=in_repo.tree().subject_ids,
            visit_ids=in_repo.tree().visit_ids,
            fill_tree=True)

        if include is None:
            include = set()
            for base in study_class.__mro__:
                if base not in skip_bases and hasattr(base, 'add_data_specs'):
                    include.update(s.name for s in base.add_data_specs
                                   if s.name not in skip)
        elif skip:
            raise BananaUsageError(
                "Cannot provide both 'include' and 'skip' options to "
                "PipelineTester.generate_test_data")

        # Generate all derived data
        for spec_name in sorted(include):
            study.data(spec_name)

        # Get output repository to write the data to
        if out_server is not None:
            out_repo = XnatRepo(project_id=out_repo, server=out_server,
                                cache_dir=op.join(work_dir, 'xnat-cache'))
        else:
            out_repo = BasicRepo(out_repo, depth=repo_depth)

        # Upload data to repository
        for spec in study.data_specs():
            try:
                data = study.data(spec.name, generate=False)
            except ArcanaMissingDataException:
                continue
            for item in data:
                if not item.exists:
                    logger.info("Skipping upload of non-existant {}"
                                .format(item))
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


def resolve_class(class_str):
    parts = class_str.split('.')
    module_name = '.'.join(parts[:-1])
    class_name = parts[-1]
    module = import_module(module_name)
    return getattr(module, class_name)


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
                        help=("Base classes of which to skip data specs from"))
    parser.add_argument('--reprocess', action='store_true', default=False,
                        help=("Whether to reprocess previously generated "
                              "datasets in the output repository"))
    parser.add_argument('--repo_depth', type=int, default=0,
                        help="The depth of the input repository")
    args = parser.parse_args()

    logger = logging.getLogger('nipype.workflow')
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Get Study class
    study_class = resolve_class(args.study_class)

    skip_bases = [resolve_class(c) for c in args.skip_base]

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
        parameters=parameters, skip=args.skip, skip_bases=skip_bases,
        reprocess=args.reprocess, repo_depth=args.repo_depth)


if __name__ == '__main__':
    from banana.study.mri import MriStudy
#     from banana.study.mri.t1 import T1Study
# 
#     PipelineTester.generate_test_data(
#         T1Study, '/Users/tclose/Data/t1', 'TESTBANANAT1',
#         in_server=None, out_server='https://mbi-xnat.erc.monash.edu.au',
#         work_dir='/Users/tclose/Data/t1-work',
#         skip=['t2_coreg'],
#         skip_bases=[MriStudy],
#         include=None,
#         reprocess=False, repo_depth=1)

    from banana.study.mri.dwi import DwiStudy

    PipelineTester.generate_test_data(
        DwiStudy, '/Users/tclose/Data/dwi', 'TESTBANANADWI',
        in_server=None, out_server='https://mbi-xnat.erc.monash.edu.au',
        work_dir='/Users/tclose/Data/dwi-work',
        skip=['dwi_reference', 'coreg_ref_wmseg', 'field_map_mag',
              'field_map_phase', 'moco', 'align_mats', 'moco_par',
              'field_map_delta_te', 'norm_intensity',
              'norm_intens_fa_template', 'norm_intens_wm_mask'],
        skip_bases=[MriStudy],
        parameters={
            'num_global_tracks': int(1e6)}, include=None,
        reprocess=False, repo_depth=1)
