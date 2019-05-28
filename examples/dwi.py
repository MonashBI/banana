#!/usr/bin/env python3
import os.path as op
from arcana import (
    InputFilesets, BasicRepo, SingleProc, StaticEnv)
from banana.study.mri.dwi import DwiStudy
from banana.file_format import dicom_format

study = DwiStudy(
    name='example_diffusion',
    repository=BasicRepo(
        op.join(op.expanduser('~'), 'Downloads', 'test-dir'), depth=0),
    processor=SingleProc(work_dir=op.expanduser('~/work')),
    environment=StaticEnv(),
    inputs=[InputFilesets('magnitude', 'R_L.*', dicom_format, is_regex=True),
            InputFilesets('reverse_phase', 'L_R.*', dicom_format,
                            is_regex=True)],
    parameters={'num_global_tracks': int(1e6)})

# Generate whole brain tracks and return path to cached dataset
wb_tcks = study.data('global_tracks')
for sess_tcks in wb_tcks:
    print("Performed whole-brain tractography for {}:{} session, the results "
          "are stored at '{}'"
          .format(sess_tcks.subject_id, sess_tcks.visit_id, sess_tcks.path))
