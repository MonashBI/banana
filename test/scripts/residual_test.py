#!/usr/bin/env python3
import os.path as op
from arcana import Dataset, MultiProc, StaticEnv, FilesetFilter
from banana.analysis.mri import DwiAnalysis


dataset_path = '/data_qnap/thomasc/HBA'
pe_dir = 'j'

analysis = DwiAnalysis(
    name='ISMRM',
    dataset=Dataset(dataset_path, depth=1),
    processor=MultiProc(
        num_processes=16,
        cpus_per_task=2,
        clean_work_dir_between_runs=False,
        work_dir=op.expanduser('/data_qnap/thomasc/work2'),
        prov_ignore=(
            MultiProc.DEFAULT_PROV_IGNORE
            + ['workflow/nodes/.*/parameters/nthreads'])),
    environment=StaticEnv(),
    inputs=[FilesetFilter('series', '.*DTI.*', is_regex=True)],
    enforce_inputs=False,
    parameters={'pe_dir': pe_dir,
                'response_algorithm': 'dhollander',
                'residual_method': 'ss3t_csd'})
                # 'response_algorithm': 'tax',
                # 'residual_method': 'odf'})

# Generate whole brain tracks and return path to cached dataset
residual = analysis.data(
    'residual_thresholded', derive=True,
    subject_ids=[
        'ABC_0003_IG_LTFU2', 'ABC_0005_KO_LTFU2B', 'ABC_0009_LC_LTFU3',
        'ABC_0016_AS_LTFU2B', 'ABC_0024_DK_LTFU2B', 'ABC_0030_JN_LTFU1',
        'ABC_0006_BO_LTFU1', 'ABC_0009_LC_LTFU3B', 'ABC_0020_PD_LTFU1'])

for f in residual:
    print("Residual created at {}".format(f.path))
