from banana.study.mri.t2star import T2starStudy
from banana.utils.testing import StudyTester, TEST_CACHE_DIR
from banana import InputFilesets
from arcana.repository.xnat import XnatRepo


class TestMriBase(StudyTester):

    name = 'default'
    study_class = T2starStudy
    parameters = {
        'mni_tmpl_resolution': 1}
    inputs = ['magnitude', 'coreg_ref']
