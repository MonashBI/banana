import logging  # @IgnorePep8
import os.path as op
from nipype import config
config.enable_debug_mode()
from arcana import InputFilesets, SingleProc  # @IgnorePep8
# from banana.testing import BaseTestCase as TestCase  # @IgnorePep8 @Reimport
from arcana import MultiStudy, MultiStudyMetaClass, SubStudySpec, Parameter  # @IgnorePep8
from banana.file_format import zip_format, dicom_format  # @IgnorePep8
from banana.study.mri.structural.t1 import T1Study  # @IgnorePep8
from banana.study.mri.structural.t2star import T2starStudy  # @IgnorePep8


logger = logging.getLogger('arcana')

test_data = '/Users/tclose/Data/qsm-test'

single_echo_dir = op.join(test_data, 'single-echo')
double_echo_dir = op.join(test_data, 'double-echo')


class T2StarT1Study(MultiStudy, metaclass=MultiStudyMetaClass):

    add_substudy_specs = [
        SubStudySpec(
            't1', T1Study),
        SubStudySpec(
            't2star', T2starStudy,
            name_map={'t1_brain': 'coreg_ref_brain',
                      't1_coreg_to_tmpl_ants_mat': 'coreg_to_tmpl_ants_mat',
                      't1_coreg_to_tmpl_ants_warp': 'coreg_to_tmpl_ants_warp'})]


study = T2StarT1Study(
    'qsm_corrected_times',
    repository=single_echo_dir,
    processor=SingleProc(op.join(test_data, 'work')),
    inputs=[
        InputFilesets('t2star_channels', 'swi_coils_icerecon', zip_format),
        InputFilesets('t2star_header_image', 'SWI_Images', dicom_format),
        InputFilesets('t2star_swi', 'SWI_Images', dicom_format),
        InputFilesets('t1_magnitude', dicom_format,
                        't1_mprage_sag_p2_iso_1mm')],
    parameters=[
        Parameter('t2star_reorient_to_std', False),
        Parameter('t1_reorient_to_std', False)])

# print(study.data('t2star_channel_mags', clean_work_dir=True).path(
#     subject_id='SUBJECT', visit_id='VISIT'))

print(study.data('t2star_vein_mask', clean_work_dir=True).path(
    subject_id='SUBJECT', visit_id='VISIT'))
