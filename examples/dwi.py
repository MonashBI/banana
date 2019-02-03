#!/usr/bin/env python3
import os.path as op
from arcana import FilesetSelector, DirectoryRepository, SlurmProcessor
from banana.study.mri.dwi import DwiStudy
from banana.file_format import dicom_format

study = DwiStudy(
    name='example_diffusion',
    directory=DirectoryRepository(
        op.join(op.expanduser('~'), 'Downloads', 'test-dir')),
    processor=SlurmProcessor(work_dir=op.expanduser('~/work')),
    inputs={'primary': FilesetSelector('R-L ep2d_diff.*', dicom_format,
                                       is_regex=True),
            'reverse_phase': FilesetSelector('L-R ep2d_diff.*', dicom_format,
                                             is_regex=True)},
    parameters={'num_wb_tracks': 1e8, 'toolchain': 'mrtrix'})

# Generate whole brain tracks and return path to cached dataset
wb_tcks = study.data('whole_brain_tracks')
for sess_tcks in wb_tcks:
    print("Performed whole-brain tractography for {}:{} session, the results "
          "are stored at '{}'"
          .format(sess_tcks.subject_id, sess_tcks.visit_id, sess_tcks.path))
