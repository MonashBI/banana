import os
import os.path as op
import logging
import tempfile
import shutil
from copy import copy
from pprint import pformat
from itertools import chain
from unittest import TestCase
from arcana.exceptions import ArcanaNameError
from arcana import (InputFilesets, InputFields, BasicRepo, XnatRepo, SingleProc,
                    Field, Fileset, ModulesEnv, StaticEnv)
from arcana.data.spec import BaseInputSpec
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

USE_MODULES = 'BANANA_TEST_USE_MODULES' in os.environ


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
    environment = ModulesEnv() if USE_MODULES else StaticEnv()

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
                inpt = InputFilesets(
                    spec.name, spec.name, repository=self.ref_repo)
            else:
                inpt = InputFields(
                    spec.name, spec.name, dtype=spec.dtype,
                    repository=self.ref_repo)
            # Check whether a corresponding data exists in the reference repo
            try:
                inpt.match(self.ref_repo.cached_tree(),
                           valid_formats=getattr(spec, 'valid_formats', None))
            except ArcanaInputMissingMatchError:
                continue
            self.inputs[spec.name] = inpt
        # Create the reference study
        self.ref_study = self.study_class(
            self.name,
            repository=self.ref_repo,
            processor=self.work_dir,
            inputs=self.inputs.values(),
            environment=self.environment,
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
            environment=self.environment,
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
                           include_bases=(), reprocess=False, repo_depth=0,
                           modules_env=False, clean_work_dir=True,
                           loggers=('nipype.workflow', 'arcana', 'banana')):
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
        include_bases : list[type(Study)]
            List of base classes in which all entries in their data
            specification are added to the list to include
        reprocess : bool
            Whether to reprocess the generated datasets
        repo_depth : int
            The depth of the input repository
        modules_env : bool
            Whether to use modules environment or not
        clean_work_dir : bool
            Whether to clean the Nipype work directory or not
        """

        for logger_name in loggers:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)

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
                    inpt = InputFilesets(item.basename, item.basename,
                                         item.format, repository=in_repo)
                else:
                    inpt = InputFields(item.name, item.name, item.dtype,
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

        if modules_env:
            env = ModulesEnv()
        else:
            env = StaticEnv()

        study = study_class(
            study_name,
            repository=temp_repo,
            processor=SingleProc(
                work_dir, reprocess=reprocess,
                clean_work_dir_between_runs=clean_work_dir,
                prov_ignore=(
                    SingleProc.DEFAULT_PROV_IGNORE +
                    ['.*/pkg_version',
                     'workflow/nodes/.*/requirements/.*'])),
            environment=env,
            inputs=inputs,
            parameters=parameters,
            subject_ids=in_repo.tree().subject_ids,
            visit_ids=in_repo.tree().visit_ids,
            fill_tree=True)

        if include is None:
            # Get set of methods that could override pipeline getters in
            # base classes that are not included
            potentially_overridden = set()
            for cls in chain(include_bases, [study_class]):
                potentially_overridden.update(cls.__dict__.keys())

            include = set()
            for base in study_class.__mro__:
                if not hasattr(base, 'add_data_specs'):
                    continue
                for spec in base.add_data_specs:
                    if isinstance(spec, BaseInputSpec) or spec.name in skip:
                        continue
                    if (base is study_class or base in include_bases or
                            spec.pipeline_getter in potentially_overridden):
                        include.add(spec.name)

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
                                .format(item.name))
                    continue
                if skip is not None and item.name in skip:
                    logger.info("Forced skip of {}".format(item.name))
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
                logger.info("Uploading {}".format(item_cpy))
                item_cpy.put()
                logger.info("Uploaded {}".format(item_cpy))
        logger.info("Finished generating and uploading test data for {}"
                    .format(study_class))


if __name__ == '__main__':
    from banana.study.mri.base import MriStudy
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('generate', nargs='+', help="datasets to generate")
    parser.add_argument('--data_dir', default=op.expanduser('~/Data'),
                        help="The data dir to use")
    parser.add_argument('--dont_clean_work_dir', action='store_true',
                        default=False, help="Don't clean work dir")
    args = parser.parse_args()

    print("Generating test data for {}".format(args.generate))

    if 'mri' in args.generate:

        PipelineTester.generate_test_data(
            MriStudy, op.join(args.data_dir, 'mri'), 'TESTBANANAMRI',
            in_server=None, out_server='https://mbi-xnat.erc.monash.edu.au',
            work_dir=op.join(args.data_dir, 'mri-work'),
            reprocess=False, repo_depth=0, modules_env=True,
            skip=['channels', 'mag_channels', 'phase_channels'],
            clean_work_dir=(not args.dont_clean_work_dir),
            parameters={'mni_template_resolution': 1})

    if 'mri2' in args.generate:

        PipelineTester.generate_test_data(
            MriStudy, op.join(args.data_dir, 'mri2'), 'TESTBANANAMRI2',
            in_server=None, out_server='https://mbi-xnat.erc.monash.edu.au',
            work_dir=op.join(args.data_dir, 'mri2-work'),
            reprocess=False, repo_depth=0, modules_env=True,
            clean_work_dir=(not args.dont_clean_work_dir),
            include=['brain_coreg'],
            skip=['template', 'template_brain', 'template_mask'],
            parameters={
                'coreg_method': 'flirt'})

    if 'base3' in args.generate:

        PipelineTester.generate_test_data(
            MriStudy, op.join(args.data_dir, 'mri'), 'TESTBANANAMRI3',
            in_server=None, out_server='https://mbi-xnat.erc.monash.edu.au',
            work_dir=op.join(args.data_dir, 'mri3-work'),
            reprocess=False, repo_depth=0, modules_env=True,
            clean_work_dir=(not args.dont_clean_work_dir),
            include=['brain_coreg'])

    if 'bold' in args.generate:
        from banana.study.mri.bold import BoldStudy

        PipelineTester.generate_test_data(
            BoldStudy, op.join(args.data_dir, 'bold'), 'TESTBANANABOLD',
            in_server=None, out_server='https://mbi-xnat.erc.monash.edu.au',
            work_dir=op.join(args.data_dir, 'bold-work'),
            reprocess=False, repo_depth=0, modules_env=True,
            clean_work_dir=(not args.dont_clean_work_dir),
            skip=['field_map_delta_te', 'cleaned_file'],
            parameters={
                'mni_template_resolution': 2})

    if 't1' in args.generate:
        from banana.study.mri.t1 import T1Study

        PipelineTester.generate_test_data(
            T1Study, op.join(args.data_dir, 't1'), 'TESTBANANAT1',
            in_server=None, out_server='https://mbi-xnat.erc.monash.edu.au',
            work_dir=op.join(args.data_dir, 't1-work'),
            skip=['t2_coreg'],
            include=None,
            reprocess=False, repo_depth=1, modules_env=True,
            clean_work_dir=(not args.dont_clean_work_dir))

    if 't2' in args.generate:
        from banana.study.mri.t2 import T2Study

        PipelineTester.generate_test_data(
            T2Study, op.join(args.data_dir, 't2'), 'TESTBANANAT2',
            in_server=None, out_server='https://mbi-xnat.erc.monash.edu.au',
            work_dir=op.join(args.data_dir, 't2-work'),
            reprocess=False, repo_depth=0, modules_env=True,
            clean_work_dir=(not args.dont_clean_work_dir))

    if 't2star' in args.generate:
        from banana.study.mri.t2star import T2starStudy

        PipelineTester.generate_test_data(
            T2starStudy, op.join(args.data_dir, 't2star'), 'TESTBANANAT2S',
            in_server=None, out_server='https://mbi-xnat.erc.monash.edu.au',
            work_dir=op.join(args.data_dir, 't2star-work'),
            reprocess=False, repo_depth=0, modules_env=True,
            parameters={
                'mni_template_resolution': 2},
            clean_work_dir=(not args.dont_clean_work_dir))

    if 'dwi' in args.generate:
        from banana.study.mri.dwi import DwiStudy
        from banana.study.mri.epi import EpiSeriesStudy

        PipelineTester.generate_test_data(
            DwiStudy, op.join(args.data_dir, 'dwi'), 'TESTBANANADWI',
            in_server=None, out_server='https://mbi-xnat.erc.monash.edu.au',
            work_dir=op.join(args.data_dir, 'dwi-work'),
            skip=['coreg_ref_wmseg', 'field_map_mag', 'brain_coreg',
                  'field_map_phase', 'moco', 'align_mats', 'moco_par',
                  'field_map_delta_te', 'norm_intensity', 'brain_mask_coreg',
                  'norm_intens_fa_template', 'norm_intens_wm_mask',
                  'connectome'],
            include_bases=[EpiSeriesStudy],
            parameters={
                'num_global_tracks': int(1e6)}, include=None,
            reprocess=False, repo_depth=1, modules_env=True,
            clean_work_dir=(not args.dont_clean_work_dir))

    if 'dwi2' in args.generate:
        from banana.study.mri.dwi import DwiStudy  # @Reimport

        PipelineTester.generate_test_data(
            DwiStudy, op.join(args.data_dir, 'dwi2'), 'TESTBANANADWI2',
            in_server=None, out_server='https://mbi-xnat.erc.monash.edu.au',
            work_dir=op.join(args.data_dir, 'dwi2-work'),
            include=['wm_odf'],
            reprocess=False, repo_depth=1, modules_env=True,
            clean_work_dir=(not args.dont_clean_work_dir))

    if 'dwi3' in args.generate:
        from banana import MultiStudy, MultiStudyMetaClass, SubStudySpec
        from banana.study.mri.dwi import DwiStudy  # @Reimport
        from banana.study.mri.t1 import T1Study  # @Reimport

        class DwiT1Study(MultiStudy, metaclass=MultiStudyMetaClass):

            add_substudy_specs = [
                SubStudySpec(
                    't1',
                    T1Study,
                    name_map={
                        'coreg_ref': 'dwi_mag_preproc'}),
                SubStudySpec(
                    'dwi',
                    DwiStudy,
                    name_map={
                        'anat_5tt': 't1_five_tissue_type',
                        'anat_fs_recon_all': 't1_fs_recon_all'})]

        PipelineTester.generate_test_data(
            DwiT1Study, op.join(args.data_dir, 'dwi3'), 'TESTBANANADWI3',
            in_server=None, out_server='https://mbi-xnat.erc.monash.edu.au',
            work_dir=op.join(args.data_dir, 'dwi3-work'),
            include=['dwi_connectome'],
            reprocess=False, repo_depth=0, modules_env=True,
            clean_work_dir=(not args.dont_clean_work_dir))
