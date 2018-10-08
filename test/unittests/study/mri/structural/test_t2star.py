import logging  # @IgnorePep8
import os.path as op
from nipype import config
config.enable_debug_mode()
from arcana import FilesetMatch, LinearProcessor  # @IgnorePep8
# from nianalysis.testing import BaseTestCase as TestCase  # @IgnorePep8 @Reimport

from nianalysis.file_format import zip_format, dicom_format  # @IgnorePep8
from nianalysis.study.mri.structural.t2star import T2StarT1Study  # @IgnorePep8

logger = logging.getLogger('arcana')

test_data = '/Users/tclose/Data/qsm-test'

single_echo_dir = op.join(test_data, 'single-echo')
double_echo_dir = op.join(test_data, 'double-echo')


study = T2StarT1Study(
    'qsm_new',
    repository=single_echo_dir,
    processor=LinearProcessor(op.join(test_data, 'work')),
    inputs=[
        FilesetMatch('t2star_coil_channels', zip_format, 'swi_coils_icerecon'),
        FilesetMatch('t2star_header_image', dicom_format, 'SWI_Images'),
        FilesetMatch('t2star_swi', dicom_format, 'SWI_Images'),
        FilesetMatch('t1_magnitude', dicom_format,
                     't1_mprage_sag_p2_iso_1mm')])

print(study.data('vein_mask', clean_work_dir=False).path(subject_id='SUBJECT',
                                                         visit_id='VISIT'))
