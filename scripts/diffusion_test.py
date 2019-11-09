from arcana import BasicRepo, SingleProc, InputFilesets
from banana.analysis.mri.structural.diffusion import DwiAnalysis
from banana.file_format import dicom_format
import os.path as op

test_dir = op.join(op.dirname(__file__), '..', 'test', 'data',
                   'diffusion-test')

analysis = DwiAnalysis(
    'diffusion',
    BasicRepo(op.join(test_dir, 'analysis')),
    SingleProc(op.join(test_dir, 'work')),
    inputs=[InputFilesets('magnitude', dicom_format, '16.*',
                         is_regex=True),
            InputFilesets('reverse_phase', dicom_format, '15.*',
                         is_regex=True)])

print('FA: {}'.format(analysis.data('fa').path(subject_id='subject',
                                            visit_id='visit')))
print('ADC: {}'.format(analysis.data('adc').path(subject_id='subject',
                                              visit_id='visit')))
# print('tracking: {}'.format(analysis.data('wb_tracking').path))
