import os.path as op
from banana.study.mri.t2star import T2starStudy
from banana.utils.testing import StudyTester, TEST_CACHE_DIR
from banana import InputFilesets
from banana import ModulesEnv


class TestT2starDefault(StudyTester):

    study_class = T2starStudy
    inputs = ['kspace']
    dataset_name = op.join('study', 'mri', 't2star')
    parameters = {}


if __name__ == '__main__':

    TestT2starDefault().generate_reference_data(
        'channels', environment=ModulesEnv(detect_exact_versions=False))
