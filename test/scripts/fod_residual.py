import os.path as op
from arcana import Dataset, SingleProc, StaticEnv, FilesetFilter
from banana.analysis.mri import DwiAnalysis
from banana.file_format import mrtrix_image_format

shell = 'single'

# if shell == 'single':
#     dataset_path = '/Users/tclose/Data/single-shell'
#     pe_dir = 'RL'
# else:
#     dataset_path = '/Users/tclose/Data/multi-shell'


pe_dir = 'j'

analysis = DwiAnalysis(
    name='ISMRM',
    dataset=Dataset('/Users/tclose/Data/new-test/ABC_0049_RM_LTFU2', depth=0),
    processor=SingleProc(
        work_dir=op.expanduser('~/work4'),
        prov_ignore=(
            SingleProc.DEFAULT_PROV_IGNORE
            + ['workflow/nodes/.*/requirements/.*/version']),
        clean_work_dir_between_runs=True,
        reprocess=True),
    environment=StaticEnv(),
    inputs=[FilesetFilter('series', '.*DTI.*', mrtrix_image_format,
                          is_regex=True)],
    enforce_inputs=False,
    parameters={'pe_dir': pe_dir,
                'response_algorithm': 'dhollander',
                'residual_method': 'ss3t_csd'})

# print(analysis.b_shells())

# Generate whole brain tracks and return path to cached dataset
residual = analysis.data('residual', derive=True)

for f in residual:
    print(f"Residual created at {f.path}")
