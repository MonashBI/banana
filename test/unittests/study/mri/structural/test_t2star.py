import logging  # @IgnorePep8
import os.path as op
from nipype import config
config.enable_debug_mode()
from arcana import FilesetSelector, LinearProcessor  # @IgnorePep8
# from nianalysis.testing import BaseTestCase as TestCase  # @IgnorePep8 @Reimport
from arcana import MultiStudy, MultiStudyMetaClass, SubStudySpec, Parameter  # @IgnorePep8
from nianalysis.file_format import zip_format, dicom_format  # @IgnorePep8
from nianalysis.study.mri.structural.t1 import T1Study  # @IgnorePep8
from nianalysis.study.mri.structural.t2star import T2StarStudy  # @IgnorePep8


logger = logging.getLogger('arcana')

test_data = '/Users/tclose/Data/qsm-test'

single_echo_dir = op.join(test_data, 'single-echo')
double_echo_dir = op.join(test_data, 'double-echo')


class T2StarT1Study(MultiStudy, metaclass=MultiStudyMetaClass):

    add_sub_study_specs = [
        SubStudySpec(
            't1', T1Study),
        SubStudySpec(
            't2star', T2StarStudy,
            name_map={'t1_brain': 'coreg_ref_brain',
                      't1_coreg_to_atlas_mat': 'coreg_to_atlas_mat',
                      't1_coreg_to_atlas_warp': 'coreg_to_atlas_warp'})]


study = T2StarT1Study(
    'qsm_corrected_times',
    repository=single_echo_dir,
    processor=LinearProcessor(op.join(test_data, 'work')),
    inputs=[
        FilesetSelector('t2star_channels', zip_format, 'swi_coils_icerecon'),
        FilesetSelector('t2star_header_image', dicom_format, 'SWI_Images'),
        FilesetSelector('t2star_swi', dicom_format, 'SWI_Images'),
        FilesetSelector('t1_magnitude', dicom_format,
                        't1_mprage_sag_p2_iso_1mm')],
    parameters=[
        Parameter('t2star_reorient_to_std', False),
        Parameter('t1_reorient_to_std', False)])

# print(study.data('t2star_channel_mags', clean_work_dir=True).path(
#     subject_id='SUBJECT', visit_id='VISIT'))

print(study.data('t2star_vein_mask', clean_work_dir=True).path(
    subject_id='SUBJECT', visit_id='VISIT'))
