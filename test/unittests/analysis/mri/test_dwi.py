import os.path as op
from arcana import SingleProc
from banana.analysis.mri.dwi import DwiAnalysis
from banana.utils.testing import AnalysisTester, TEST_CACHE_DIR
from banana import FilesetFilter
from banana import ModulesEnv


class TestDwi(AnalysisTester):

    analysis_class = DwiAnalysis
    inputs = ['series', 'reverse_phase']
    dataset_name = op.join('analysis', 'mri', 'dwi')
    parameters = {'num_global_tracks': int(1e5)}
    skip_specs = ['channels', 'brain_coreg', 'coreg_fsl_mat', 
                  'coreg_ants_mat', 'mag_coreg',
                  'brain_mask_coreg', 'norm_intensity',
                  'norm_intens_fa_template', 'norm_intens_wm_mask',
                  'series_coreg', 'mag_coreg_to_tmpl',
                  'coreg_to_tmpl_fsl_coeff', 'coreg_to_tmpl_fsl_report',
                  'coreg_to_tmpl_ants_mat', 'coreg_to_tmpl_ants_warp',
                  'grad_dirs_coreg', 'connectome', 'motion_mats',
                  'moco', 'align_mats', 'moco_par', 'qformed', 'qform_mat',
                  'field_map_delta_te']
    


if __name__ == '__main__':

    from banana.entrypoint import set_loggers
    from nipype import config
    config.enable_debug_mode()

    set_loggers([('banana', 'INFO'), ('arcana', 'INFO'),
                 ('nipype.workflow', 'INFO')])

    TestDwi().generate_reference_data(
        environment=ModulesEnv(detect_exact_versions=False),
        prov_ignore=SingleProc.DEFAULT_PROV_IGNORE + [
            'workflow/nodes/.*/requirements/.*/version'],
        clean_work_dir_between_runs=False,
        reprocess=True)
