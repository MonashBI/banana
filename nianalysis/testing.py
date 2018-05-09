import os.path
import nianalysis
from arcana.testing import (
    BaseTestCase as NianalysisTestCase,
    BaseMultiSubjectTestCase as NianalysisMultiSubjectTestCase)

MBIANALSYSIS_BASE_TEST_DIR = os.path.abspath(os.path.join(
    os.path.dirname(nianalysis.__file__), '..', 'test'))


class BaseTestCase(NianalysisTestCase):

    BASE_TEST_DIR = MBIANALSYSIS_BASE_TEST_DIR


class BaseMultiSubjectTestCase(NianalysisMultiSubjectTestCase):

    BASE_TEST_DIR = MBIANALSYSIS_BASE_TEST_DIR
