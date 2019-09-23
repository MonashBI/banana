from banana.study.mri.t1w import T1wStudy
from banana.utils.testing import StudyTester, TEST_CACHE_DIR
from arcana.repository.xnat import XnatRepo


class TestT1wStudy(StudyTester):

    name = 'T1'
    study_class = T1wStudy
    ref_repo = XnatRepo(server='https://mbi-xnat.erc.monash.edu.au',
                        project_id='TESTBANANAT1',
                        cache_dir=TEST_CACHE_DIR)

    parameters = {'aparc_atlas': 'DKT'}

    def test_aparc_stats_table_pipeline(self):
        self.run_pipeline_test('aparc_stats_table_pipeline',
                               pipeline_args={'measure': 'thickness',
                                              'hemisphere': 'rh'})
