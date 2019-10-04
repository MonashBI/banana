import os.path as op
from arcana import SingleProc
from banana.study.mri.t2star import T2starStudy
from banana.utils.testing import StudyTester, TEST_CACHE_DIR
from banana import InputFilesets
from banana import ModulesEnv


class TestT2star(StudyTester):

    study_class = T2starStudy
    inputs = ['kspace']
    dataset_name = op.join('study', 'mri', 't2star')
    parameters = {}
    skip_specs = ['mag_coreg', 'brain_coreg', 'brain_mask_coreg']


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
