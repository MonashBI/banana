#!/usr/bin/env python3
import os.path as op
from arcana import (
    FilesetFilter, Dataset, SingleProc, StaticEnv)
from banana.analysis.mri.dwi import DwiAnalysis
from banana.file_format import dicom_format

analysis = DwiAnalysis(
    name='example_diffusion',
    dataset=Dataset(
        op.join(op.expanduser('~'), 'Downloads', 'test-dir'), depth=0),
    processor=SingleProc(work_dir=op.expanduser('~/work')),
    environment=StaticEnv(),
    inputs=[FilesetFilter('magnitude', 'R_L.*', dicom_format, is_regex=True),
            FilesetFilter('reverse_phase', 'L_R.*', dicom_format,
                          is_regex=True)],
    parameters={'num_global_tracks': int(1e6)})

# Generate whole brain tracks and return path to cached dataset
wb_tcks = analysis.data('global_tracks', derive=True)
for sess_tcks in wb_tcks:
    print("Performed whole-brain tractography for {}:{} session, the results "
          "are stored at '{}'"
          .format(sess_tcks.subject_id, sess_tcks.visit_id, sess_tcks.path))
