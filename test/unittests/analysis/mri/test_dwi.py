from banana.analysis.mri.dwi import DwiAnalysis
from banana.utils.testing import AnalysisTester, TEST_CACHE_DIR
from arcana.repository.xnat import XnatRepo


class TestMriAnalysis(AnalysisTester):

    name = 'Dwi'
    analysis_class = DwiAnalysis
    ref_repo = XnatRepo(server='https://mbi-xnat.erc.monash.edu.au',
                        project_id='TESTBANANADWI',
                        cache_dir=TEST_CACHE_DIR)

    def test_preprocess_channels_pipeline(self):
        pass  # Need to upload some raw channel data for this

#     def test_linear_coreg_pipeline(self):
#         self.run_pipeline_test('linear_coreg_pipeline')
