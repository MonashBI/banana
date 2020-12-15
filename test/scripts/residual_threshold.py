#!/usr/bin/env python3
import os.path as op
from arcana import Dataset, SingleProc, StaticEnv, FilesetFilter
from banana.analysis.mri import DwiAnalysis
from banana.file_format import nifti_gz_format


dataset_path = op.expanduser('~/Data/residual-output')
pe_dir = 'j'

analysis = DwiAnalysis(
    name='residual',
    dataset=Dataset(dataset_path, depth=1),
    processor=SingleProc(
        work_dir=op.expanduser('~/data/work4')),
    environment=StaticEnv(),
    inputs=[FilesetFilter('residual', 'tensor',
                          valid_formats=nifti_gz_format)],
    enforce_inputs=False,
    parameters={'pe_dir': pe_dir,
                'response_algorithm': 'tax',
                'residual_method': 'odf'})

# Generate whole brain tracks and return path to cached dataset
residual = analysis.data('residual_thresholded', derive=True)

for f in residual:
    print("Residual created at {}".format(f.path))
