from arcana import BasicRepo, SingleProc, InputFileset
from banana.study.mri.structural.diffusion import DwiStudy
from banana.file_format import dicom_format
import os.path as op

test_dir = op.join(op.dirname(__file__), '..', 'test', 'data',
                   'diffusion-test')

study = DwiStudy(
    'diffusion',
    BasicRepo(op.join(test_dir, 'study')),
    SingleProc(op.join(test_dir, 'work')),
    inputs=[InputFileset('magnitude', dicom_format, '16.*',
                         is_regex=True),
            InputFileset('reverse_phase', dicom_format, '15.*',
                         is_regex=True)])

print('FA: {}'.format(study.data('fa').path(subject_id='subject',
                                            visit_id='visit')))
print('ADC: {}'.format(study.data('adc').path(subject_id='subject',
                                              visit_id='visit')))
# print('tracking: {}'.format(study.data('wb_tracking').path))
