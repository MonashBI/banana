#!/usr/bin/env python3
import os.path as op
from arcana import Dataset, MultiProc, StaticEnv, FilesetFilter
from banana.analysis.mri import DwiAnalysis

shell = 'multi'

dataset_path = op.expanduser('~/data/QA-test')
pe_dir = 'j'

analysis = DwiAnalysis(
    name='residual',
    dataset=Dataset(dataset_path, depth=1),
    processor=MultiProc(
        work_dir=op.expanduser('~/data/work')),
    environment=StaticEnv(),
    inputs=[FilesetFilter('series', 'dwi')],
    enforce_inputs=False,
    parameters={'pe_dir': pe_dir,
                'response_algorithm': 'tax',
                'residual_method': 'odf'})

# Generate whole brain tracks and return path to cached dataset
residual = analysis.data('residual', derive=True)

for f in residual:
    print("Residual created at {}".format(f.path))

