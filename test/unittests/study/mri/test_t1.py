from banana.study.mri.t1 import T1Study
from banana.utils.testing import PipelineTester, TEST_CACHE_DIR
from arcana.repository.xnat import XnatRepo


class TestMriStudy(PipelineTester):

    name = 'T1'
    study_class = T1Study
    ref_repo = XnatRepo(server='https://mbi-xnat.erc.monash.edu.au',
                        project_id='TESTBANANAT1',
                        cache_dir=TEST_CACHE_DIR)

    parameters = {'aparc_atlas': 'desikan-killiany-tourville'}

    def test_aparc_stats_table_pipeline(self):
        self.run_pipeline_test('aparc_stats_table_pipeline',
                               pipeline_args={'measure': 'thickness',
                                              'hemisphere': 'rh'})
