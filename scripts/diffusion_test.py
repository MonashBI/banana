from arcana import LocalFileSystemRepo, SingleProc, FilesetFilter
from banana.analysis.mri import DwiAnalysis
from banana.file_format import dicom_format
import os.path as op

test_dir = op.join(op.dirname(__file__), '..', 'test', 'data',
                   'diffusion-test')

analysis = DwiAnalysis(
    'diffusion',
    LocalFileSystemRepo(op.join(test_dir, 'analysis')),
    SingleProc(op.join(test_dir, 'work')),
    inputs=[FilesetFilter('magnitude', dicom_format, '16.*',
                          is_regex=True),
            FilesetFilter('reverse_phase', dicom_format, '15.*',
                          is_regex=True)])

print('FA: {}'.format(analysis.derive('fa', derive=True).path(subject_id='subject',
                                               visit_id='visit')))
print('ADC: {}'.format(analysis.derive('adc', derive=True).path(subject_id='subject',
                                                 visit_id='visit')))
# print('tracking: {}'.format(analysis.derive('wb_tracking').path))
