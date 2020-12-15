#!/usr/bin/env python3
import os.path as op
from arcana import Dataset, SingleProc, StaticEnv, FilesetFilter
from banana.analysis.mri import DwiAnalysis
from banana.file_format import nifti_gz_format


dataset_path = op.expanduser('~/Data/residual-output')
pe_dir = 'j'

analysis = DwiAnalysis(
    name='derivatives',
    dataset=Dataset(
        dataset_path,
        depth=1,
        subject_ids=[
            'ABC_0003_IG_LTFU2', 'ABC_0005_KO_LTFU2B', 'ABC_0009_LC_LTFU3',
            'ABC_0016_AS_LTFU2B', 'ABC_0024_DK_LTFU2B', 'ABC_0030_JN_LTFU1',
            'ABC_0006_BO_LTFU1', 'ABC_0009_LC_LTFU3B', 'ABC_0020_PD_LTFU1']),
    processor=SingleProc(
        work_dir=op.expanduser('~/Data/work5')),
    environment=StaticEnv(),
    inputs=[FilesetFilter('residual', 'residual-tensor',
                          valid_formats=nifti_gz_format),
            FilesetFilter('series_preproc', 'series_preproc',
                          valid_formats=nifti_gz_format),
            FilesetFilter('grad_dirs', 'grad_dirs'),
            FilesetFilter('bvalues', 'bvalues')],
    enforce_inputs=False,
    parameters={'pe_dir': pe_dir,
                'response_algorithm': 'tax',
                'residual_method': 'odf'})

# Generate whole brain tracks and return path to cached dataset
residual, comb_mag = analysis.data(
    ['residual_thresholded', 'combined_magnitude'], derive=True)

for f in residual:
    print("Residual created at {}".format(f.path))
