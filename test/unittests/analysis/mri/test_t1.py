from banana.analysis.mri.t1w import T1wAnalysis
from banana.utils.testing import AnalysisTester, TEST_CACHE_DIR
from arcana.repository.xnat import XnatRepo


class TestT1wAnalysis(AnalysisTester):

    name = 'T1'
    analysis_class = T1wAnalysis
    ref_repo = XnatRepo(server='https://mbi-xnat.erc.monash.edu.au',
                        project_id='TESTBANANAT1',
                        cache_dir=TEST_CACHE_DIR)

    parameters = {'aparc_atlas': 'DKT'}

    def test_aparc_stats_table_pipeline(self):
        self.run_pipeline_test('aparc_stats_table_pipeline',
                               pipeline_args={'measure': 'thickness',
                                              'hemisphere': 'rh'})
