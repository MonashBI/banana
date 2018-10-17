import os.path
import subprocess as sp
import shutil
from unittest import TestCase
import errno
from xnat.exceptions import XNATError
import sys
import json
import filecmp
from collections import defaultdict
import warnings
import logging
import xnat
from arcana.repository.xnat import (
    XnatRepository, special_char_re, lower)
from arcana.exception import ArcanaMissingDataException
from arcana.utils import split_extension
import banana
from arcana.utils import classproperty
from arcana.processor import LinearProcessor
from arcana.exception import ArcanaError
from arcana.node import ArcanaNodeMixin
from arcana.exception import (
    ArcanaModulesNotInstalledException)
from traceback import format_exc
from arcana.repository.directory import (
    DirectoryRepository, SUMMARY_NAME,
    SUMMARY_NAME as LOCAL_SUMMARY_NAME, FIELDS_FNAME)

logger = logging.getLogger('arcana')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

logging.getLogger("urllib3").setLevel(logging.WARNING)


class BaseTestCase(TestCase):

    SUBJECT = 'SUBJECT'
    VISIT = 'VISIT'
    SERVER = 'https://mbi-xnat.erc.monash.edu.au'
    XNAT_TEST_PROJECT = 'TEST001'
    REF_SUFFIX = '_REF'

    # The path to the test directory, which should sit along side the
    # the package directory. Note this will not work when Arcana
    # is installed by a package manager.
    BASE_TEST_DIR = os.path.abspath(os.path.join(
        os.path.dirname(banana.__file__), '..', 'test'))

    @classproperty
    @classmethod
    def test_data_dir(cls):
        return os.path.join(cls.BASE_TEST_DIR, 'data')

    @classproperty
    @classmethod
    def unittest_root(cls):
        return os.path.join(cls.BASE_TEST_DIR, 'unittests')

    @classproperty
    @classmethod
    def repository_path(cls):
        return os.path.join(cls.test_data_dir, 'repository')

    @classproperty
    @classmethod
    def work_path(cls):
        return os.path.join(cls.test_data_dir, 'work')

    @classproperty
    @classmethod
    def base_cache_path(cls):
        return os.path.join(cls.test_data_dir, 'cache')

    def setUp(self, cache_dir=None):
        self.reset_dirs()
        self.add_session(self.project_dir, self.SUBJECT, self.VISIT,
                         cache_dir=cache_dir)

    def add_session(self, project_dir, subject, session,
                    required_filesets=None, cache_dir=None):
        if cache_dir is None:
            cache_dir = self.cache_dir
        session_dir = os.path.join(project_dir, subject, session)
        os.makedirs(session_dir)
        try:
            download_all_filesets(
                cache_dir, self.SERVER, self.xnat_session_name,
                overwrite=False)
        except Exception:
            if os.path.exists(cache_dir):
                warnings.warn(
                    "Could not download filesets from '{}_{}' session on "
                    "MBI-XNAT, attempting with what has already been "
                    "downloaded:\n\n{}"
                    .format(self.XNAT_TEST_PROJECT, self.name, format_exc()))
            else:
                raise
        for f in os.listdir(cache_dir):
            if required_filesets is None or f in required_filesets:
                src_path = os.path.join(cache_dir, f)
                dst_path = os.path.join(session_dir, f)
                if os.path.isdir(src_path):
                    shutil.copytree(src_path, dst_path)
                elif os.path.isfile(src_path):
                    shutil.copy(src_path, dst_path)
                else:
                    assert False
        # Download reference
        try:
            download_all_filesets(
                cache_dir, self.SERVER,
                self.xnat_session_name + self.REF_SUFFIX,
                overwrite=False)
        except Exception:
            pass

    def delete_project(self, project_dir):
        # Clean out any existing repository files
        shutil.rmtree(project_dir, ignore_errors=True)

    def reset_dirs(self):
        shutil.rmtree(self.project_dir, ignore_errors=True)
        shutil.rmtree(self.work_dir, ignore_errors=True)
        self.create_dirs()

    def create_dirs(self):
        for d in (self.project_dir, self.work_dir, self.cache_dir):
            if not os.path.exists(d):
                os.makedirs(d)

    @property
    def xnat_session_name(self):
        return '{}_{}'.format(self.XNAT_TEST_PROJECT, self.name)

    @property
    def session_dir(self):
        return self.get_session_dir(self.SUBJECT, self.VISIT)

    @property
    def cache_dir(self):
        return os.path.join(self.base_cache_path, self.name)

    @property
    def repository(self):
        return DirectoryRepository(self.project_dir)

    @property
    def processor(self):
        return LinearProcessor(self.work_dir)

    @property
    def project_dir(self):
        return os.path.join(self.repository_path, self.name)

    @property
    def work_dir(self):
        return os.path.join(self.work_path, self.name)

    @property
    def name(self):
        return self._get_name(type(self))

    @property
    def project_id(self):
        return self.name  # To allow override in deriving classes

    def _get_name(self, cls):
        """
        Get unique name for test class from module path and its class name to
        be used for storing test data on XNAT and creating unique work/project
        dirs
        """
        module_path = os.path.abspath(sys.modules[cls.__module__].__file__)
        rel_module_path = module_path[(len(self.unittest_root) + 1):]
        path_parts = rel_module_path.split(os.path.sep)
        module_name = (''.join(path_parts[:-1]) +
                       os.path.splitext(path_parts[-1])[0][5:]).upper()
        test_class_name = cls.__name__[4:].upper()
        return module_name + '_' + test_class_name

    def create_study(self, study_cls, name, inputs, repository=None,
                     processor=None, **kwargs):
        """
        Creates a study using default repository and processors.

        Parameters
        ----------
        study_cls : Study
            The class to initialise
        name : str
            Name of the study
        inputs : List[BaseSpec]
            List of inputs to the study
        repository : BaseRepository | None
            The repository to use (a default local repository is used if one
            isn't provided
        processor : Processor | None
            The processor to use (a default LinearProcessor is used if one
            isn't provided
        """
        if repository is None:
            repository = self.repository
        if processor is None:
            processor = self.processor
        return study_cls(
            name=name,
            repository=repository,
            processor=processor,
            inputs=inputs,
            **kwargs)

    def assertFilesetCreated(self, fileset_name, study_name, subject=None,
                             visit=None, frequency='per_session'):
        output_dir = self.get_session_dir(subject, visit, frequency)
        out_path = self.output_file_path(
            fileset_name, study_name, subject, visit, frequency)
        self.assertTrue(
            os.path.exists(out_path),
            ("Fileset '{}' (expected at '{}') was not created by unittest"
             " ('{}' found in '{}' instead)".format(
                 fileset_name, out_path, "', '".join(os.listdir(output_dir)),
                 output_dir)))

    def assertField(self, name, ref_value, study_name, subject=None,
                    visit=None, frequency='per_session',
                    to_places=None):
        esc_name = study_name + '_' + name
        output_dir = self.get_session_dir(subject, visit, frequency)
        try:
            with open(os.path.join(output_dir, FIELDS_FNAME)) as f:
                fields = json.load(f)
        except OSError as e:
            if e.errno == errno.ENOENT:
                raise ArcanaError(
                    "No fields were created by pipeline in study '{}'"
                    .format(study_name))
        try:
            value = fields[esc_name]
        except KeyError:
            raise ArcanaError(
                "Field '{}' was not created by pipeline in study '{}'. "
                "Created fields were ('{}')"
                .format(esc_name, study_name, "', '".join(fields)))
        msg = ("Field value '{}' for study '{}', {}, does not match "
               "reference value ({})".format(name, study_name, value,
                                             ref_value))
        if to_places is not None:
            self.assertAlmostEqual(
                value, ref_value, to_places,
                '{} to {} decimal places'.format(msg, to_places))
        else:
            self.assertEqual(value, ref_value, msg)

    def assertFilesetsEqual(self, fileset1, fileset2, error_msg=None):
        msg = "{} does not match {}".format(fileset1, fileset2)
        if msg is not None:
            msg += ':\n' + error_msg
        self.assertTrue(filecmp.cmp(fileset1.path, fileset2.path,
                                    shallow=False), msg=msg)

#     def assertImagesMatch(self, output, ref, study_name):
#         out_path = self.output_file_path(output, study_name)
#         ref_path = self.ref_file_path(ref)
#         try:
#             sp.check_output('diff {}.nii {}.nii'
#                             .format(out_path, ref_path), shell=True)
#         except sp.CalledProcessError as e:
#             if e.output == "Binary files {} and {} differ\n".format(
#                     out_path, ref_path):
#                 self.assert_(
#                     False,
#                     "Images {} and {} do not match exactly".format(out_path,
#                                                                    ref_path))
#             else:
#                 raise

    def assertStatEqual(self, stat, fileset_name, target, study_name,
                        subject=None, visit=None,
                        frequency='per_session'):
            try:
                ArcanaNodeMixin.load_module('mrtrix')
            except ArcanaModulesNotInstalledException:
                pass
            val = float(sp.check_output(
                'mrstats {} -output {}'.format(
                    self.output_file_path(
                        fileset_name, study_name,
                        subject=subject, visit=visit,
                        frequency=frequency),
                    stat),
                shell=True))
            self.assertEqual(
                val, target, (
                    "{} value of '{}' ({}) does not equal target ({}) "
                    "for subject {} visit {}"
                    .format(stat, fileset_name, val, target,
                            subject, visit)))

    def assertImagesAlmostMatch(self, out, ref, mean_threshold,
                                stdev_threshold, study_name):
        out_path = self.output_file_path(out, study_name)
        ref_path = self.ref_file_path(ref)
        # Should probably look into ITK fuzzy matching methods
        cmd = ("mrcalc -quiet {a} {b} -subtract - | mrstats - | "
               "grep -v channel | awk '{{print $4 \" \" $6}}'"
               .format(a=out_path, b=ref_path))
        out = sp.check_output(cmd, shell=True)
        mean, stdev = (float(x) for x in out.split())
        self.assertTrue(
            abs(mean) < mean_threshold and stdev < stdev_threshold,
            ("Mean ({mean}) or standard deviation ({stdev}) of difference "
             "between images {a} and {b} differ more than threshold(s) "
             "({thresh_mean} and {thresh_stdev} respectively)"
             .format(mean=mean, stdev=stdev, thresh_mean=mean_threshold,
                     thresh_stdev=stdev_threshold, a=out_path, b=ref_path)))

    def get_session_dir(self, subject=None, visit=None,
                        frequency='per_session'):
        if subject is None and frequency in ('per_session', 'per_subject'):
            subject = self.SUBJECT
        if visit is None and frequency in ('per_session', 'per_visit'):
            visit = self.VISIT
        if frequency == 'per_session':
            assert subject is not None
            assert visit is not None
            path = os.path.join(self.project_dir, subject, visit)
        elif frequency == 'per_subject':
            assert subject is not None
            assert visit is None
            path = os.path.join(
                self.project_dir, subject, SUMMARY_NAME)
        elif frequency == 'per_visit':
            assert visit is not None
            assert subject is None
            path = os.path.join(self.project_dir, SUMMARY_NAME, visit)
        elif frequency == 'per_study':
            assert subject is None
            assert visit is None
            path = os.path.join(self.project_dir, SUMMARY_NAME, SUMMARY_NAME)
        else:
            assert False
        return os.path.abspath(path)

    @classmethod
    def remove_generated_files(cls, study=None):
        # Remove derived filesets
        for fname in os.listdir(cls.get_session_dir()):
            if study is None or fname.startswith(study + '_'):
                os.remove(os.path.join(cls.get_session_dir(), fname))

    def output_file_path(self, fname, study_name, subject=None, visit=None,
                         frequency='per_session', **kwargs):
        return os.path.join(
            self.get_session_dir(subject=subject, visit=visit,
                                 frequency=frequency, **kwargs),
            '{}_{}'.format(study_name, fname))

    def ref_file_path(self, fname, subject=None, session=None):
        return os.path.join(self.session_dir, fname,
                            subject=subject, session=session)


class BaseMultiSubjectTestCase(BaseTestCase):

    SUMMARY_NAME = LOCAL_SUMMARY_NAME

    def setUp(self, cache_dir=None):
        self.reset_dirs()
        self.add_sessions(self.project_dir, cache_dir=cache_dir)

    def add_sessions(self, project_dir, required_filesets=None,
                     cache_dir=None):
        if cache_dir is None:
            cache_dir = self.cache_dir
        try:
            download_all_filesets(
                cache_dir, self.SERVER, self.xnat_session_name,
                overwrite=False)
        except XNATError as e:
            if os.path.exists(cache_dir):
                warnings.warn(
                    "Could not download filesets from '{}_{}' session on "
                    "MBI-XNAT, attempting with what has already been "
                    "downloaded:\n\n{}"
                    .format(self.XNAT_TEST_PROJECT, self.name, e))
            else:
                raise
        fnames = os.listdir(cache_dir)
        for fname in fnames:
            if fname == FIELDS_FNAME:
                continue
            if fname.startswith('.'):
                continue
            subj_id, visit_id, fileset = self._extract_ids(fname)
            if required_filesets is None or fileset in required_filesets:
                session_dir = self.make_session_dir(
                    project_dir, subj_id, visit_id)
                src_path = os.path.join(cache_dir, fname)
                dst_path = os.path.join(session_dir, fileset)
                if os.path.isdir(src_path):
                    shutil.copytree(src_path, dst_path)
                elif os.path.isfile(src_path):
                    shutil.copy(src_path, dst_path)
                else:
                    assert False
        if FIELDS_FNAME in fnames:
            fields = defaultdict(lambda: defaultdict(dict))
            with open(os.path.join(cache_dir, FIELDS_FNAME), 'rb') as f:
                all_fields = json.load(f)
            for name, value in list(all_fields.items()):
                subj_id, visit_id, field_name = self._extract_ids(name)
                fields[subj_id][visit_id][field_name] = value
            for subj_id, subj_fields in list(fields.items()):
                for visit_id, visit_fields in list(subj_fields.items()):
                    session_dir = self.make_session_dir(
                        project_dir, subj_id, visit_id)
                    with open(os.path.join(session_dir, FIELDS_FNAME),
                              'w') as f:
                        json.dump(visit_fields, f)

    def _extract_ids(self, name):
        parts = name.split('_')
        if len(parts) < 3:
            raise ArcanaError(
                "'{}' in multi-subject test session '{}' needs to be "
                "prepended with subject and session IDs (delimited by "
                "'_')".format(name, self.xnat_session_name))
        subj_id, visit_id = parts[:2]
        if subj_id.lower() == SUMMARY_NAME.lower():
            subj_id = SUMMARY_NAME
        if visit_id.lower() == SUMMARY_NAME.lower():
            visit_id = SUMMARY_NAME
        basename = '_'.join(parts[2:])
        return subj_id, visit_id, basename

    @property
    def subject_ids(self):
        return (d for d in os.listdir(self.project_dir)
                if d != self.SUMMARY_NAME)

    def visit_ids(self, subject_id):
        subject_dir = os.path.join(self.project_dir, subject_id)
        return (d for d in os.listdir(subject_dir)
                if d != self.SUMMARY_NAME)

    def session_dir(self, subject, visit):
        return self.get_session_dir(subject, visit)

    def get_session_dir(self, subject, visit, frequency='per_session'):
        return super(BaseMultiSubjectTestCase, self).get_session_dir(
            subject=subject, visit=visit, frequency=frequency)

    @classmethod
    def make_session_dir(cls, project_dir, subj_id, visit_id):
        session_dir = os.path.join(project_dir, subj_id,
                                   visit_id)
        try:
            os.makedirs(session_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        return session_dir


class DummyTestCase(BaseTestCase):

    def __init__(self):
        self.setUp()

    def __del__(self):
        self.tearDown()

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def assert_(self, statement, message=None):
        if not statement:
            message = "'{}' is not true".format(statement)
            print(message)
        else:
            print("Test successful")

    def assertEqual(self, first, second, message=None):
        if first != second:
            if message is None:
                message = '{} and {} are not equal'.format(repr(first),
                                                           repr(second))
            print(message)
        else:
            print("Test successful")

    def assertAlmostEqual(self, first, second, message=None):
        if first != second:
            if message is None:
                message = '{} and {} are not equal'.format(repr(first),
                                                           repr(second))
            print(message)
        else:
            print("Test successful")

    def assertLess(self, first, second, message=None):
        if first >= second:
            if message is None:
                message = '{} is not less than {}'.format(repr(first),
                                                          repr(second))
            print(message)
        else:
            print("Test successful")


class TestTestCase(BaseTestCase):

    def test_test(self):
        pass


def download_all_filesets(download_dir, server, session_id, overwrite=True,
                          **kwargs):
    with xnat.connect(server, **kwargs) as xnat_login:
        try:
            session = xnat_login.experiments[session_id]
        except KeyError:
            raise ArcanaMissingDataException(
                "Didn't find session matching '{}' on {}"
                .format(session_id, server))
        try:
            os.makedirs(download_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        for fileset in session.scans.values():
            file_format = XnatRepository.guess_file_format(fileset)
            ext = file_format.extension
            if ext is None:
                ext = ''
            download_path = os.path.join(download_dir, fileset.type + ext)
            if overwrite or not os.path.exists(download_path):
                download_resource(download_path, fileset,
                                  file_format, session.label)
        fields = {}
        for name, value in list(session.fields.items()):
            # Try convert to each datatypes in order of specificity to
            # determine type
            fields[name] = value
        with open(os.path.join(download_dir, FIELDS_FNAME), 'w') as f:
            json.dump(fields, f)


def download_fileset(download_path, server, user, password, session_id,
                     fileset_name, file_format=None):
    """
    Downloads a single fileset from an XNAT server
    """
    with xnat.connect(server, user=user, password=password) as xnat_login:
        try:
            session = xnat_login.experiments[session_id]
        except KeyError:
            raise ArcanaError(
                "Didn't find session matching '{}' on {}".format(session_id,
                                                                 server))
        try:
            fileset = session.scans[fileset_name]
        except KeyError:
            raise ArcanaError(
                "Didn't find fileset matching '{}' in {}".format(fileset_name,
                                                                 session_id))
        if file_format is None:
            file_format = XnatRepository.guess_file_format(fileset)
        download_resource(download_path, fileset, file_format,
                          session.label)


def download_resource(download_path, fileset, file_format,
                      session_label):
    xresource = None
    for resource_name in file_format.xnat_resource_names:
        try:
            xresource = fileset.resources[resource_name]
            break
        except KeyError:
            logger.debug(
                "Did not find resource corresponding to '{}' for {}, "
                "will try alternatives if available".format(
                    resource_name, fileset))
            continue
    if xresource is None:
        raise ArcanaError(
            "Didn't find '{}' resource(s) in {} fileset matching "
            "'{}' in {}".format(
                "', '".join(file_format.xnat_resource_names),
                fileset.type))
    tmp_dir = download_path + '.download'
    xresource.download_dir(tmp_dir)
    fileset_label = fileset.id + '-' + special_char_re.sub('_', fileset.type)
    src_path = os.path.join(tmp_dir, session_label, 'scans',
                            fileset_label, 'resources', xresource.label,
                            'files')
    if not file_format.directory:
        fnames = os.listdir(src_path)
        match_fnames = [
            f for f in fnames
            if lower(split_extension(f)[-1]) == lower(
                file_format.extension)]
        if len(match_fnames) == 1:
            src_path = os.path.join(src_path, match_fnames[0])
        else:
            raise ArcanaMissingDataException(
                "Did not find single file with extension '{}' "
                "(found '{}') in resource '{}'"
                .format(file_format.extension,
                        "', '".join(fnames), src_path))
    shutil.move(src_path, download_path)
    shutil.rmtree(tmp_dir)


def list_filesets(server, user, password, session_id):
    with xnat.connect(server, user=user, password=password) as xnat_login:
        session = xnat_login.experiments[session_id]
        return [s.type for s in session.scans.values()]
