from banana.study.mri.base import MriStudy
from banana.utils.testing import PipelineTester, TEST_CACHE_DIR
from arcana.repository.xnat import XnatRepo


class TestMriStudy(PipelineTester):

    name = 'BaseMri'
    study_class = MriStudy
    ref_repo = XnatRepo(server='https://mbi-xnat.erc.monash.edu.au',
                        project_id='TESTBANANAMRI',
                        cache_dir=TEST_CACHE_DIR)

    def test_preprocess_channels_pipeline(self):
        pass  # Need to upload some raw channel data for this

    def test_linear_coreg_pipeline(self):
        self.run_pipeline_test('linear_coreg_pipeline')

    def test_brain_extraction_pipeline(self):
        self.run_pipeline_test('brain_extraction_pipeline')

    def test_brain_coreg_pipeline(self):
        self.run_pipeline_test('brain_coreg_pipeline',
                               add_inputs=['coreg_ref'])

    def test_coreg_matrix_pipeline(self):
        self.run_pipeline_test('coreg_matrix_pipeline',
                               add_inputs=['coreg_ref'])

    def test_coreg_to_tmpl_pipeline(self):
        self.run_pipeline_test('coreg_to_tmpl_pipeline',
                               add_inputs=['coreg_ref'],
                               test_criteria={
                                   'coreg_to_tmpl': {'rms_tol': 20000}})

    def test_qform_transform_pipeline(self):
        self.run_pipeline_test('qform_transform_pipeline',
                               add_inputs=['coreg_ref'])

    def test_preprocess_pipeline(self):
        self.run_pipeline_test('preprocess_pipeline')

    def test_header_extraction_pipeline(self):
        self.run_pipeline_test('header_extraction_pipeline')

    def test_motion_mat_pipeline(self):
        self.run_pipeline_test('motion_mat_pipeline')
