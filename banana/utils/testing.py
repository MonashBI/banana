
import os
import os.path as op
import logging
import tempfile
import shutil
from copy import copy
from pprint import pformat
from itertools import chain
from unittest import TestCase
from nipype.pipeline.plugins import DebugPlugin
from arcana.exceptions import ArcanaNameError
from arcana import (FilesetFilter, FieldFilter, XnatRepo,
                    Field, Fileset, ModulesEnv, StaticEnv, SingleProc,
                    MultiProc)
from arcana.repository import LocalFileSystemRepo, Dataset
from arcana.data.spec import BaseInputSpecMixin
from arcana.processor.base import Processor
from arcana.exceptions import (
    ArcanaInputMissingMatchError, ArcanaMissingDataException,
    ArcanaReprocessException)
from arcana.utils.testing import BaseTestCase  # pylint: disable=unused-import
import banana
from banana.exceptions import BananaTestSetupError, BananaUsageError


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

try:
    TEST_DATA_ROOT = os.environ['BANANA_TEST_DATA_ROOT']
except KeyError:
    # Use the path if the repository has been checked out
    TEST_DATA_ROOT = op.join(op.dirname(banana.__file__), '..', 'test',
                             'ref-data')

USE_MODULES = 'BANANA_TEST_USE_MODULES' in os.environ

if USE_MODULES:
    TEST_ENV = ModulesEnv()
else:
    TEST_ENV = StaticEnv()


TEST_CACHE_DIR = op.join(TEST_DIR, 'cache')


class DontRunProc(Processor):
    """
    A thin wrapper around the NiPype DebugPlugin that doesn't run any nodes
    """

    nipype_plugin_cls = DebugPlugin

    def __init__(self):
        tmp_dir = tempfile.mkdtemp()
        super().__init__(work_dir=tmp_dir, reprocess=False,
                         callable=self.dont_run)

    @classmethod
    def dont_run(cls, node, graph):
        pass


class AnalysisTester(TestCase):

    required_atttrs = (
        ('analysis_class', 'class of the analysis to test'),
        ('dataset_name', 'select dataset to use for inputs'),
        ('inputs', 'inputs for the analysis'),
        ('parameters', 'parameters for the analysis'))

    def runTest(self):
        for attr, reason in self.required_atttrs:
            if not hasattr(self, attr):
                raise BananaTestSetupError(
                    "{} class doesn't have '{}' class attribute, reqquired"
                    "to {}".format(self.__class__.__name__, attr, reason))
        analysis = self.analysis_class(  # pylint: disable=no-member
            name=self.name,  # pylint: disable=no-member
            inputs=self.inputs_dict,
            parameters=self.parameters,  # pylint: disable=no-member
            dataset=self.dataset,
            processor=DontRunProc())

        all_pipelines = set(
            analysis.spec(n).pipeline for n in analysis.data_spec_names())

        try:
            analysis.run(*all_pipelines)
        except ArcanaReprocessException as e:
            raise self.failureException(self._formatMessage(
                str(e), ("Provenance of reference data no longer matches that "
                         "of generated pipelines for {} test of {} analysis class"
                         .format(self.name, self.analysis_class.__name__))))  # noqa pylint: disable=no-member

    @property
    def name(self):
        return type(self).__name__

    @property
    def dataset(self):
        return Dataset(op.join(TEST_DATA_ROOT, self.dataset_name))  # noqa pylint: disable=no-member

    @property
    def inputs_dict(self):
        return {i: i for i in self.inputs}  # pylint: disable=no-member

    def generate_reference_data(self, *spec_names, processor=None,
                                work_dir=None, environment=None, **kwargs):
        """
        Generates reference data and provenance against which the unittests
        are run against
        """
        if work_dir is None:
            work_dir = tempfile.mkdtemp()
        if processor is None:
            processor = SingleProc(work_dir=work_dir, **kwargs)
        if environment is None:
            environment = StaticEnv()
        analysis = self.analysis_class(  # pylint: disable=no-member
            name=self.name,  # pylint: disable=no-member
            inputs=self.inputs_dict,
            parameters=self.parameters,  # pylint: disable=no-member
            dataset=self.dataset,
            environment=environment,
            processor=processor)
        if not spec_names:
            try:
                skip_specs = self.skip_specs
            except AttributeError:
                skip_specs = ()
            spec_names = [s.name for s in analysis.data_specs()
                          if s.derived and s.name not in skip_specs]
        analysis.derive(spec_names)


class PipelineTester(TestCase):
    """
    Runs pipelines within a Analysis class and compares the results
    against reference data from previous runs

    Class attributes
    ----------------
    name : str
        A unique name (among other unittests to be run in same batch) for the
        test class
    analysis_class : type(Analysis)
        The analysis class run tests against
    ref_dataset : Dataset
        The dataset to draw the reference data from
    parameters : dict[str, str | float | int | datetime]
        The parameters passed to the analysis when it is initialised
    warn_missing_tests : bool
        Whether to raise a warning if there is a pipeline in the analysis class
        that does not have a test in tester class
    """

    parameters = {}
    warn_missing_tests = True
    environment = ModulesEnv() if USE_MODULES else StaticEnv()
    name = None
    analysis_class = None
    ref_dataset = None

    def setUp(self):
        if not hasattr(self, 'analysis_class'):
            raise BananaTestSetupError(
                "{} must have a 'analysis_class' attribute corresponding to the "
                "Analysis class to test".format(type(self).__name__))
        if not hasattr(self, 'ref_dataset'):
            raise BananaTestSetupError(
                "{} must have a 'ref_dataset' attribute corresponding to the "
                "Analysis class to test".format(type(self).__name__))
        base_test_dir = op.join(TEST_DIR, self.name)
        self.work_dir = op.join(base_test_dir, 'work')
        dataset_root = op.join(base_test_dir, 'dataset')
        os.makedirs(self.work_dir, exist_ok=True)
        os.makedirs(dataset_root, exist_ok=True)
        # Create datasetsitory to hold outputs
        self.output_dataset = Dataset(dataset_root, depth=2)
        # Create inputs for reference analysis
        self.inputs = {}
        for spec in self.analysis_class.data_specs():
            # Create an input for each entry in the class specificiation
            if spec.is_fileset:
                inpt = FilesetFilter(
                    spec.name, spec.name, dataset=self.ref_dataset)
            else:
                inpt = FieldFilter(
                    spec.name, spec.name, dtype=spec.dtype,
                    dataset=self.ref_dataset)
            # Check whether a corresponding data exists in the reference dataset
            try:
                inpt.match(self.ref_dataset.tree,
                           valid_formats=getattr(spec, 'valid_formats', None))
            except ArcanaInputMissingMatchError:
                continue
            self.inputs[spec.name] = inpt
        # Create the reference analysis
        self.ref_analysis = self.analysis_class(  # pylint: disable=not-callable
            self.name,
            repository=self.ref_dataset,
            processor=self.work_dir,
            inputs=self.inputs.values(),
            environment=self.environment,
            parameters=self.parameters)
        # Get set of all pipelines to test
        if self.warn_missing_tests:
            pipelines_to_test = set(
                s.pipeline_getter for s in self.analysis_class.data_specs()
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
            Inputs that are required in the output analysis for the pipeline to
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
        # A analysis with all inputs provided to determine which inputs are needed
        # by the pipeline
        ref_pipeline = self.ref_analysis.pipeline(pipeline_getter,
                                                  pipeline_args=pipeline_args)
        inputs = []
        for spec_name in chain(ref_pipeline.input_names, add_inputs):
            try:
                inputs.append(self.inputs[spec_name])
            except KeyError:
                pass  # Inputs with a default value
        # Set up output analysis
        output_analysis = self.analysis_class(  # pylint: disable=not-callable
            pipeline_getter,
            datasetsitory=self.output_dataset,
            processor=SingleProc(self.work_dir, reprocess='force'),
            environment=self.environment,
            inputs=inputs,
            parameters=self.parameters,
            subject_ids=self.ref_analysis.subject_ids,
            visit_ids=self.ref_analysis.visit_ids,
            enforce_inputs=False,
            fill_tree=True)
        for spec_name in ref_pipeline.output_names:
            for ref, test in zip(self.ref_analysis.derive(spec_name),
                                 output_analysis.derive(spec_name)):
                if ref.is_fileset:
                    try:
                        self.assertTrue(
                            test.contents_equal(
                                ref, **test_criteria.get(spec_name, {})),
                            "'{}' fileset generated by {} in {} doesn't match "
                            "reference".format(spec_name, pipeline_getter,
                                               self.analysis_class))
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
                            self.analysis_class, ref.value))

    @classmethod
    def generate_test_data(cls, analysis_class, in_dataset, out_dataset,
                           in_server=None, out_server=None, work_dir=None,
                           parameters=(), include=None, skip=(),
                           include_bases=(), reprocess=False, dataset_depth=0,
                           modules_env=False, clean_work_dir=True,
                           loggers=('nipype.workflow', 'arcana', 'banana')):
        """
        Generates reference data for a pipeline tester unittests given a analysis
        class and set of parameters

        Parameters
        ----------
        analysis_class : type(Analysis)
            The path to the analysis class to test, e.g. banana.analysis.MriAnalysis
        in_dataset : str
            The path to dataset that houses the input data
        out_dataset : str
            If the 'xnat_server' argument is provided then out
            is interpreted as the project ID to use the XNAT
            server (the project must exist already). Otherwise
            it is interpreted as the path to a basic dataset
        in_server : str | None
            The server to download the input data from
        out_server : str | None
            The server to upload the reference data to
        work_dir : str
            The work directory
        parameters : dict[str, *]
            Parameter to set when initialising the analysis
        include : list[str] | None
            Spec names to include in the output dataset. If None all names
            except those listed in 'skip' are included
        skip : list[str]
            Spec names to skip in the generation process. Only valid if
            'include' is None
        include_bases : list[type(Analysis)]
            List of base classes in which all entries in their data
            specification are added to the list to include
        reprocess : bool
            Whether to reprocess the generated datasets
        dataset_depth : int
            The depth of the input dataset
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

        if analysis_class.__name__.endswith('Analysis'):
            analysis_name = analysis_class.__name__[:-len('Analysis')]
        else:
            analysis_name = analysis_class.__name__

        # Get output dataset to write the data to
        if in_server is not None:
            in_dataset = XnatRepo(
                server=in_server,
                cache_dir=op.join(work_dir, 'xnat-cache')).dataset(in_dataset)
        else:
            in_dataset = Dataset(in_dataset, depth=dataset_depth)

        temp_dataset_root = op.join(work_dir, 'temp-dataset')
        if os.path.exists(temp_dataset_root) and reprocess:
            shutil.rmtree(temp_dataset_root)
        os.makedirs(temp_dataset_root, exist_ok=True)

        temp_dataset = Dataset(temp_dataset_root, depth=dataset_depth)

        inputs = None
        for session in in_dataset.tree().sessions:
            session_inputs = []
            for item in chain(session.filesets, session.fields):
                if isinstance(item, Fileset):
                    inpt = FilesetFilter(item.basename, item.basename,
                                         item.format, dataset=in_dataset)
                else:
                    inpt = FieldFilter(item.name, item.name, item.dtype,
                                       dataset=in_dataset)
                try:
                    spec = analysis_class.data_spec(inpt)
                except ArcanaNameError:
                    print("Skipping {} as it doesn't match a spec in {}"
                          .format(item, analysis_class))
                else:
                    session_inputs.append(inpt)
            session_inputs = sorted(session_inputs)
            if inputs is not None and session_inputs != inputs:
                raise BananaUsageError(
                    "Inconsistent inputs ({} and {}) found in sessions of {}"
                    .format(inputs, session_inputs, in_dataset))
            else:
                inputs = session_inputs

        if modules_env:
            env = ModulesEnv()
        else:
            env = StaticEnv()

        analysis = analysis_class(
            analysis_name,
            dataset=temp_dataset,
            processor=SingleProc(
                work_dir, reprocess=reprocess,
                clean_work_dir_between_runs=clean_work_dir,
                prov_ignore=(
                    SingleProc.DEFAULT_PROV_IGNORE
                    + ['.*/pkg_version',
                       'workflow/nodes/.*/requirements/.*'])),
            environment=env,
            inputs=inputs,
            parameters=parameters,
            subject_ids=in_dataset.tree().subject_ids,
            visit_ids=in_dataset.tree().visit_ids,
            fill_tree=True)

        if include is None:
            # Get set of methods that could override pipeline getters in
            # base classes that are not included
            potentially_overridden = set()
            for cls in chain(include_bases, [analysis_class]):
                potentially_overridden.update(cls.__dict__.keys())

            include = set()
            for base in analysis_class.__mro__:
                if not hasattr(base, 'add_data_specs'):
                    continue
                for spec in base.add_data_specs:
                    if isinstance(spec,
                                  BaseInputSpecMixin) or spec.name in skip:
                        continue
                    if (base is analysis_class or base in include_bases
                            or spec.pipeline_getter in potentially_overridden):
                        include.add(spec.name)

        # Generate all derived data
        for spec_name in sorted(include):
            analysis.derive(spec_name)

        # Get output dataset to write the data to
        if out_server is not None:
            out_dataset = XnatRepo(
                server=out_server,
                cache_dir=op.join(work_dir, 'xnat-cache')).dataset(out_dataset),
        else:
            out_dataset = Dataset(out_dataset, depth=dataset_depth)

        # Upload data to dataset
        for spec in analysis.data_specs():
            try:
                data = analysis.derive(spec.name, generate=False)
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
                        dataset=out_dataset, exists=True)
                else:
                    item_cpy = Field(
                        name=item.name, value=item.value, dtype=item.dtype,
                        frequency=item.frequency, array=item.array,
                        subject_id=item.subject_id, visit_id=item.visit_id,
                        dataset=out_dataset, exists=True)
                logger.info("Uploading {}".format(item_cpy))
                item_cpy.put()
                logger.info("Uploaded {}".format(item_cpy))
        logger.info("Finished generating and uploading test data for {}"
                    .format(analysis_class))


if __name__ == '__main__':
    from banana.analysis.mri.base import MriAnalysis
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
            MriAnalysis, op.join(args.data_dir, 'mri'), 'TESTBANANAMRI',
            in_server=None, out_server='https://mbi-xnat.erc.monash.edu.au',
            work_dir=op.join(args.data_dir, 'mri-work'),
            reprocess=False, dataset_depth=0, modules_env=True,
            skip=['channels', 'channels_dir'],
            clean_work_dir=(not args.dont_clean_work_dir),
            parameters={'mni_tmpl_resolution': 1})

    if 'mri2' in args.generate:

        PipelineTester.generate_test_data(
            MriAnalysis, op.join(args.data_dir, 'mri2'), 'TESTBANANAMRI2',
            in_server=None, out_server='https://mbi-xnat.erc.monash.edu.au',
            work_dir=op.join(args.data_dir, 'mri2-work'),
            reprocess=False, dataset_depth=0, modules_env=True,
            clean_work_dir=(not args.dont_clean_work_dir),
            include=['brain_coreg'],
            skip=['template', 'template_brain', 'template_mask'],
            parameters={
                'coreg_method': 'flirt'})

    if 'base3' in args.generate:

        PipelineTester.generate_test_data(
            MriAnalysis, op.join(args.data_dir, 'mri'), 'TESTBANANAMRI3',
            in_server=None, out_server='https://mbi-xnat.erc.monash.edu.au',
            work_dir=op.join(args.data_dir, 'mri3-work'),
            reprocess=False, dataset_depth=0, modules_env=True,
            clean_work_dir=(not args.dont_clean_work_dir),
            include=['brain_coreg'])

    if 'bold' in args.generate:
        from banana.analysis.mri.bold import BoldAnalysis

        PipelineTester.generate_test_data(
            BoldAnalysis, op.join(args.data_dir, 'bold'), 'TESTBANANABOLD',
            in_server=None, out_server='https://mbi-xnat.erc.monash.edu.au',
            work_dir=op.join(args.data_dir, 'bold-work'),
            reprocess=False, dataset_depth=0, modules_env=True,
            clean_work_dir=(not args.dont_clean_work_dir),
            skip=['field_map_delta_te', 'cleaned_file'],
            parameters={
                'mni_tmpl_resolution': 2})

    if 't1' in args.generate:
        from banana.analysis.mri.t1w import T1wAnalysis

        PipelineTester.generate_test_data(
            T1wAnalysis, op.join(args.data_dir, 't1'), 'TESTBANANAT1',
            in_server=None, out_server='https://mbi-xnat.erc.monash.edu.au',
            work_dir=op.join(args.data_dir, 't1-work'),
            skip=['t2_coreg'],
            include=None,
            reprocess=False, dataset_depth=1, modules_env=True,
            clean_work_dir=(not args.dont_clean_work_dir))

    if 't2' in args.generate:
        from banana.analysis.mri.t2w import T2wAnalysis

        PipelineTester.generate_test_data(
            T2wAnalysis, op.join(args.data_dir, 't2'), 'TESTBANANAT2',
            in_server=None, out_server='https://mbi-xnat.erc.monash.edu.au',
            work_dir=op.join(args.data_dir, 't2-work'),
            reprocess=False, dataset_depth=0, modules_env=True,
            clean_work_dir=(not args.dont_clean_work_dir))

    if 't2star' in args.generate:
        from banana.analysis.mri.t2star import T2starAnalysis

        PipelineTester.generate_test_data(
            T2starAnalysis, op.join(args.data_dir, 't2star'), 'TESTBANANAT2S',
            in_server=None, out_server='https://mbi-xnat.erc.monash.edu.au',
            work_dir=op.join(args.data_dir, 't2star-work'),
            reprocess=False, dataset_depth=0, modules_env=True,
            parameters={
                'mni_tmpl_resolution': 2},
            clean_work_dir=(not args.dont_clean_work_dir))

    if 'dwi' in args.generate:
        from banana.analysis.mri.dwi import DwiAnalysis
        from banana.analysis.mri.epi import EpiSeriesAnalysis

        PipelineTester.generate_test_data(
            DwiAnalysis, op.join(args.data_dir, 'dwi'), 'TESTBANANADWI',
            in_server=None, out_server='https://mbi-xnat.erc.monash.edu.au',
            work_dir=op.join(args.data_dir, 'dwi-work'),
            skip=['coreg_ref_wmseg', 'field_map_mag', 'brain_coreg',
                  'field_map_phase', 'moco', 'align_mats', 'moco_par',
                  'field_map_delta_te', 'norm_intensity', 'brain_mask_coreg',
                  'norm_intens_fa_template', 'norm_intens_wm_mask',
                  'connectome', 'series_coreg', 'grad_dirs_coreg',
                  'mag_coreg', 'motion_mats'],
            include_bases=[EpiSeriesAnalysis],
            parameters={
                'num_global_tracks': int(1e6)}, include=None,
            reprocess=False, dataset_depth=1, modules_env=True,
            clean_work_dir=(not args.dont_clean_work_dir))

    if 'dwi2' in args.generate:
        from banana.analysis.mri.dwi import DwiAnalysis  # @Reimport

        PipelineTester.generate_test_data(
            DwiAnalysis, op.join(args.data_dir, 'dwi2'), 'TESTBANANADWI2',
            in_server=None, out_server='https://mbi-xnat.erc.monash.edu.au',
            work_dir=op.join(args.data_dir, 'dwi2-work'),
            include=['wm_odf'],
            reprocess=False, dataset_depth=1, modules_env=True,
            clean_work_dir=(not args.dont_clean_work_dir))

    if 'dwi3' in args.generate:
        from banana import (MultiAnalysis, MultiAnalysisMetaClass, SubCompSpec)
        from banana.analysis.mri.dwi import DwiAnalysis  # @Reimport
        from banana.analysis.mri.t1w import T1wAnalysis  # @Reimport

        class DwiT1wAnalysis(MultiAnalysis, metaclass=MultiAnalysisMetaClass):

            add_subcomp_specs = [
                SubCompSpec(
                    't1',
                    T1wAnalysis,
                    name_map={
                        'coreg_ref': 'dwi_mag_preproc'}),
                SubCompSpec(
                    'dwi',
                    DwiAnalysis,
                    name_map={
                        'anat_5tt': 't1_five_tissue_type',
                        'anat_fs_recon_all': 't1_fs_recon_all'})]

        PipelineTester.generate_test_data(
            DwiT1wAnalysis, op.join(args.data_dir, 'dwi3'), 'TESTBANANADWI3',
            in_server=None, out_server='https://mbi-xnat.erc.monash.edu.au',
            work_dir=op.join(args.data_dir, 'dwi3-work'),
            include=['dwi_connectome'],
            reprocess=False, dataset_depth=0, modules_env=True,
            clean_work_dir=(not args.dont_clean_work_dir))
