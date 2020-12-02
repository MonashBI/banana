import os.path as op
from arcana import Dataset, SingleProc, StaticEnv, FilesetFilter
from banana.analysis.mri import DwiAnalysis
from banana.file_format import mrtrix_image_format

shell = 'multi'

if shell == 'single':
    dataset_path = '/Users/tclose/Data/single-shell'
    pe_dir = 'RL'
else:
    dataset_path = '/Users/tclose/Data/multi-shell'
    pe_dir = 'RL'

analysis = DwiAnalysis(
    name='multiresidual',
    dataset=Dataset(dataset_path, depth=0),
    processor=SingleProc(
        work_dir=op.expanduser('~/work'),
        prov_ignore=(
            SingleProc.DEFAULT_PROV_IGNORE
            + ['workflow/nodes/.*/requirements/.*/version'])),
    environment=StaticEnv(),
    inputs=[FilesetFilter('series', 'dwi', mrtrix_image_format)],
    enforce_inputs=False,
    parameters={'pe_dir': pe_dir,
                'response_algorithm': 'tax',
                'residual_method': 'odf'})

# print(analysis.b_shells())

# Generate whole brain tracks and return path to cached dataset
residual = analysis.data('residual', derive=True)

for f in residual:
    print(f"Residual created at {f.path}")
