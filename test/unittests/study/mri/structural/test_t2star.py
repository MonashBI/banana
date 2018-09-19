import logging  # @IgnorePep8
import os.path as op
from nipype import config
config.enable_debug_mode()
from arcana import LinearProcessor, DirectoryRepository  # @IgnorePep8
from arcana.data import FilesetMatch  # @IgnorePep8
# from nianalysis.testing import BaseTestCase as TestCase  # @IgnorePep8 @Reimport

from nianalysis.file_format import zip_format, dicom_format  # @IgnorePep8
from nianalysis.study.mri.structural.t2star import T2StarStudy  # @IgnorePep8

logger = logging.getLogger('arcana')

test_data = '/Users/tclose/Data/qsm'

single_echo_dir = op.join(test_data, 'single_echo')
double_echo_dir = op.join(test_data, 'double_echo')


study = T2StarStudy(
    'qsm',
    LinearProcessor(op.join(test_data, 'work')),
    DirectoryRepository(single_echo_dir),
    inputs=[
        FilesetMatch('coil_channels', zip_format, 'swi_coils'),
        FilesetMatch('header_image', dicom_format, 'SWI_Images')])

print(study.data('qsm').path(subject_id='SUBJECT', visit_id='VISIT'))
