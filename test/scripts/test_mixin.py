from mbianalysis.study.mri.motion_detection_mixin import (
    create_motion_detection_class)
import os.path
import errno
from nianalysis.archive.local import LocalArchive
# from nianalysis.archive.xnat import XNATArchive


list_t1 = ['t1_1_dicom']
list_t2 = ['t2_1_dicom', 't2_2_dicom', 't2_3_dicom', 't2_4_dicom',
           't2_5_dicom', 'fm_dicom']
list_epi = ['epi_1_dicom',]
list_dwi = [['dwi_1_main_dicom', '0'], ['dwi2ref_1_opposite_dicom', '-1'],
            ['dwi2ref_1_dicom', '1']]
list_utes = ['ute_dicom']

cls, inputs = create_motion_detection_class(
    'test_mixin', 'reference_dicom', 't1', t1s=list_t1, t2s=list_t2, dmris=list_dwi,
    epis=list_epi, utes=list_utes)

WORK_PATH = os.path.join('/Users', 'fsforazz', 'Desktop',
                         'test_mc_mixin_cache')
try:
    os.makedirs(WORK_PATH)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

study = cls(
    name='test_mc_mixin',
    project_id='MMH008', archive=LocalArchive(
        '/Users/fsforazz/Desktop/test_mc_mixin'),

    inputs=inputs)
study.plot_mean_displacement_pipeline().run(
    subject_ids=['MMH008_{}'.format(i) for i in ['CON012']],
    visit_ids=['MRPT01'], work_dir=WORK_PATH)

print 'Done!'
