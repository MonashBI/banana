from banana.study.mri.t2star import T2starStudy
from banana.utils.testing import StudyTester, TEST_CACHE_DIR
from banana import InputFilesets
from arcana.repository.xnat import XnatRepo


class TestT2StarDefault(StudyTester):

    study_class = T2starStudy
    inputs = ['magnitude']
