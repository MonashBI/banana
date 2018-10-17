from arcana import DirectoryRepository, LinearProcessor, FilesetSelector
from banana.study.mri.structural.diffusion import DiffusionStudy
from banana.file_format import dicom_format
import os.path as op

test_dir = op.join(op.dirname(__file__), '..', 'test', 'data',
                   'diffusion-test')

study = DiffusionStudy(
    'diffusion',
    DirectoryRepository(op.join(test_dir, 'study')),
    LinearProcessor(op.join(test_dir, 'work')),
    inputs=[FilesetSelector('magnitude', dicom_format, '16.*',
                         is_regex=True),
            FilesetSelector('reverse_phase', dicom_format, '15.*',
                         is_regex=True)])

print('FA: {}'.format(study.data('fa').path(subject_id='subject',
                                            visit_id='visit')))
print('ADC: {}'.format(study.data('adc').path(subject_id='subject',
                                              visit_id='visit')))
# print('tracking: {}'.format(study.data('wb_tracking').path))
