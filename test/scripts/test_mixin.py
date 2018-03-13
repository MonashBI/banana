from mbianalysis.study.mri.motion_detection_mixin import (
    create_motion_detection_class)
import os.path
import errno
from nianalysis.archive.local import LocalArchive
import pydicom
import glob
import shutil
# from nianalysis.archive.xnat import XNATArchive


def prepare_mc_detection(input_dir):

    ref = None
    t1s = None
    t2s = None
    epis = None
    dmris = None
    utes = None
    umaps = None

    dcm_files = glob.glob(input_dir+'/*.dcm')
    if not dcm_files:
        dcm_files = glob.glob(input_dir+'/*.IMA')
    if not dcm_files:
        raise Exception('No DICOM files found in {}'.format(input_dir))
    os.mkdir(input_dir+'/work_dir')
    os.mkdir(input_dir+'/work_dir/work_sub_dir')
    os.mkdir(input_dir+'/work_dir/work_sub_dir/work_session_dir')
    working_dir = input_dir+'/work_dir/work_sub_dir/work_session_dir/'
    hdr = pydicom.read_file(dcm_files[0])
    name_scan = (
        str(hdr.SeriesNumber).zfill(2)+'_'+hdr.SeriesDescription)
    name_scan = name_scan.replace(" ", "_")
    scan_description = [name_scan]
    files = []
    for i, im in enumerate(dcm_files):
        hdr = pydicom.read_file(im)
        name_scan = (
            str(hdr.SeriesNumber).zfill(2)+'_'+hdr.SeriesDescription)
        name_scan = name_scan.replace(" ", "_")
        if name_scan in scan_description:
            files.append(im)
        else:
            if (os.path.isdir(
                    working_dir+scan_description[-1]) is False):
                os.mkdir(working_dir+scan_description[-1])
                for f in files:
                    shutil.copy(f, working_dir+scan_description[-1])
            files = [im]
            scan_description.append(name_scan)
        if i == len(dcm_files)-1:
            if (os.path.isdir(working_dir+scan_description[-1]) is
                    False):
                os.mkdir(working_dir+scan_description[-1])
                for f in files:
                    shutil.copy(f, working_dir+scan_description[-1])
    for i, scan in enumerate(scan_description, start=1):
        if i == 1:
            print 'Available scans: '
        print '{0} {1}'.format(i, scan)
    ref = raw_input("Please select the reference scan: ").split()
    ref_type = raw_input("Please enter the reference type (t1 or t2): ")
    t1s = raw_input("Please select the T1 weighted scans: ").split()
    epis = raw_input("Please select the T2 weighted scans: ").split()
    t2s = raw_input("Please select the EPI scans: ").split()
    dmris = raw_input("Please select the DWI scans: ").split()
    utes = raw_input("Please select the UTE scans: ").split()
    if utes:
        umaps = raw_input("Please select the umap: ").split()

    t1s = [scan_description[int(i)-1] for i in t1s]
    t2s = [scan_description[int(i)-1] for i in t2s]
    epis = [scan_description[int(i)-1] for i in epis]
    dmris = [x.split(',') for x in dmris]
    dmris = [[scan_description[int(i[0])-1], i[1]] for i in dmris]
    utes = [scan_description[int(i)-1] for i in utes]
    if utes:
        umaps = [scan_description[int(i)-1] for i in umaps]
    if len(ref) > 1:
        raise Exception('Only one reference can be provided, you selected {}.'
                        .format(len(ref)))
    ref = scan_description[int(ref[0])-1]

    print ref, t1s, t2s, epis, dmris, utes, umaps

    return ref, ref_type, t1s, epis, t2s, dmris, utes, umaps


# list_t1 = ['t1_1_dicom']
# list_t2 = ['t2_1_dicom', 't2_2_dicom', 't2_3_dicom', 't2_4_dicom',
#            't2_5_dicom', 'fm_dicom']
# list_epi = ['epi_1_dicom']
# list_dwi = [['dwi_1_main_dicom', '0'], ['dwi2ref_1_opposite_dicom', '-1'],
#             ['dwi2ref_1_dicom', '1']]
# list_utes = ['ute_dicom']

input_dir = '/Volumes/ELEMENTS/test_mc_mixin/'
ref, ref_type, t1s, epis, t2s, dmris, utes, umaps = (
    prepare_mc_detection(input_dir))

cls, inputs = create_motion_detection_class(
    'test_mixin', ref, ref_type, t1s=t1s, t2s=t2s, dmris=dmris, epis=epis,
    utes=utes, umaps=umaps)

WORK_PATH = os.path.join(input_dir,
                         'test_mc_mixin_cache')
try:
    os.makedirs(WORK_PATH)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

study = cls(
    name='work_dir',
    project_id='MMH008', archive=LocalArchive(input_dir),

    inputs=inputs)
study.plot_mean_displacement_pipeline().run(
    subject_ids=['work_sub_dir'],
    visit_ids=['work_session_dir'], work_dir=WORK_PATH)

print 'Done!'

# study = cls(
#     name='test_mc_mixin',
#     project_id='MMH008', archive=LocalArchive(
#         '/Users/fsforazz/Desktop/test_mc_mixin'),
# 
#     inputs=inputs)
# study.plot_mean_displacement_pipeline().run(
#     subject_ids=['MMH008_{}'.format(i) for i in ['CON012']],
#     visit_ids=['MRPT01'], work_dir=WORK_PATH)
# 
# print 'Done!'
