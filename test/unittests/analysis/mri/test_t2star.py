import os.path as op
from arcana import SingleProc
from banana.analysis.mri.t2star import T2starAnalysis
from banana.utils.testing import AnalysisTester, TEST_CACHE_DIR
from banana import FilesetFilter
from banana import ModulesEnv


class TestT2star(AnalysisTester):

    analysis_class = T2starAnalysis
    inputs = ['kspace', 'coreg_ref', 'swi']
    dataset_name = op.join('analysis', 'mri', 't2star')
    parameters = {
        'mni_tmpl_resolution': 1}
    # skip_specs = ['swi']


if __name__ == '__main__':

    from banana.entrypoint import set_loggers
    from nipype import config
    config.enable_debug_mode()

    set_loggers([('banana', 'INFO'), ('arcana', 'INFO'),
                 ('nipype.workflow', 'INFO')])

    TestT2star().generate_reference_data(
        environment=ModulesEnv(detect_exact_versions=False),
        prov_ignore=SingleProc.DEFAULT_PROV_IGNORE + [
            'workflow/nodes/.*/requirements/.*/version'],
        clean_work_dir_between_runs=False)
