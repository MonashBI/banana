from mbianalysis.study.mri.motion_detection_mixin import (
    create_motion_detection_class)
import os.path
import errno
from nianalysis.archive.local import LocalArchive
import pydicom
import glob
import shutil
# from nianalysis.archive.xnat import XNATArchive
from nianalysis.dataset import Dataset, Field  # @IgnorePep8
from nianalysis.data_formats import nifti_gz_format, dicom_format
from mbianalysis.study.mri.motion_detection_metaclass import MotionDetectionStudy
from nipype.workflows import dmri


def prepare_mc_detection(input_dir):

    ref = None
    t1s = None
    t2s = None
    epis = None
    dmris = None
    utes = None
    umaps = None

    dcm_files = sorted(glob.glob(input_dir+'/*.dcm'))
    if not dcm_files:
        dcm_files = sorted(glob.glob(input_dir+'/*.IMA'))
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
        if name_scan in scan_description[-1]:
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
    for i, scan in enumerate(sorted(scan_description), start=1):
        if i == 1:
            print 'Available scans: '
        print '{0} {1}'.format(i, scan)
    correct_ref = False
    while not correct_ref:
        ref = raw_input("Please select the reference scan: ").split()
        if not ref:
            print ('A reference image must be provided!')
        elif ref and len(ref) > 1:
            print ('Only one reference can be provided, you selected {}.'
                   .format(len(ref)))
        else:
            correct_ref = True

    ref = scan_description[int(ref[0])-1]
    correct_ref_type = False
    while not correct_ref_type:
        ref_type = raw_input(
            "Please enter the reference type ('t1' or 't2'): ")
        if ref_type != 't1' and ref_type != 't2':
            print ('{} is not a recognized ref_type!The available '
                   'ref_types are t1 or t2.'.format(ref_type[0]))
        else:
            correct_ref_type = True
    t1s = raw_input("Please select the T1 weighted scans: ").split()
    epis = raw_input("Please select the T2 weighted scans: ").split()
    t2s = raw_input("Please select the EPI scans: ").split()
    dmris = raw_input("In order to run dwi motion"
                      " correction, each dwi scan number must be followed by "
                      "one numeber: 0 for main dwi scan (the one with multiple"
                      " directions), 1 for a reference scan with the SAME "
                      "phase encoding direction (ped) as the main scan, -1 for"
                      " a reference scan with OPPOSITE ped. For example a "
                      "valid entry is: 10,0 11,-1. Please select the DWI "
                      "scans: ").split()
    if dmris:
        correct_dmris = False
        while not correct_dmris:
            try:
                dmris = [x.split(',') for x in dmris]
                dmris = [[scan_description[int(i[0])-1], i[1]] for i in dmris]
                correct_dmris = True
            except IndexError:
                print ('DWI scan and phase encoding direction entered in a '
                       'wrong way: {}'.format(dmris))
                dmris = raw_input(
                    "In order to run dwi motion"
                    " correction, each dwi scan number must be followed by "
                    "one numeber: 0 for main dwi scan (the one with multiple"
                    " directions), 1 for a reference scan with the SAME "
                    "phase encoding direction (ped) as the main scan, -1 for"
                    " a reference scan with OPPOSITE ped. For example a "
                    "valid entry is: 10,0 11,-1. Please select the DWI "
                    "scans: ").split()
                continue
    utes = raw_input("Please select the UTE scans: ").split()
    if utes:
        umaps = raw_input("Please select the umap: ").split()

    t1s = [scan_description[int(i)-1] for i in t1s]
    t2s = [scan_description[int(i)-1] for i in t2s]
    epis = [scan_description[int(i)-1] for i in epis]
    utes = [scan_description[int(i)-1] for i in utes]
    if utes:
        umaps = [scan_description[int(i)-1] for i in umaps]

    print ref, t1s, t2s, epis, dmris, utes, umaps

    return ref, ref_type, t1s, epis, t2s, dmris, utes, umaps


t1s = ['t1_1_dicom']
t2s = ['t2_1_dicom', 't2_2_dicom', 't2_3_dicom', 't2_4_dicom',
           't2_5_dicom', 'fm_dicom']
epis = ['epi_1_dicom']
dmris = [['dwi_1_main_dicom', '0'], ['dwi2ref_1_opposite_dicom', '-1'],
            ['dwi2ref_1_dicom', '1']]
utes = ['ute_dicom']
umaps = []
ref = 'reference_dicom'
ref_type = 't1'

# input_dir = '/Volumes/ELEMENTS/test_mc_mixin/'
input_dir = '/Users/francescosforazzini/git/mbi-analysis/test/'
# ref, ref_type, t1s, epis, t2s, dmris, utes, umaps = (
#     prepare_mc_detection(input_dir))
 
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
# inputs={
#                 'epi1_epi': Dataset('epi_1_dicom', dicom_format),
#                 't1_1_t1': Dataset('t1_1_dicom', dicom_format),
#                 't2_1_t2': Dataset('t2_1_dicom', dicom_format),
#                 't2_2_t2': Dataset('t2_2_dicom', dicom_format),
#                 't2_3_t2': Dataset('t2_3_dicom', dicom_format),
#                 't2_4_t2': Dataset('t2_4_dicom', dicom_format),
#                 't2_5_t2': Dataset('t2_5_dicom', dicom_format),
#                 'dwi_1_main_dwi_main': Dataset('dwi_1_main_dicom', dicom_format),
#                 'dwi_1_to_ref_dwi2ref_to_correct': Dataset('dwi2ref_1_dicom',
#                                                      dicom_format),
#                 'dwi_1_opposite_opposite_dwi2ref_to_correct': Dataset('dwi2ref_1_opposite_dicom',
#                                                      dicom_format),
#                 'dwi_1_main_dwi_main_ref': Dataset('dwi2ref_1_opposite_dicom',
#                                           dicom_format),
#                 'dwi_1_to_ref_dwi2ref_ref': Dataset('dwi2ref_1_opposite_dicom',
#                                               dicom_format),
#                 'dwi_1_opposite_opposite_dwi2ref_ref': Dataset('dwi2ref_1_dicom',
#                                               dicom_format),
#                 'ute_t1': Dataset('ute_dicom', dicom_format),
#                 'fm_t2': Dataset('fm_dicom', dicom_format),
#                 'ref_primary': Dataset('reference_dicom', dicom_format)}
# study = cls(
#     name='test_mixin',
#     project_id='work_dir', archive=LocalArchive(input_dir),
#  
#     inputs=inputs)
# study.plot_mean_displacement_pipeline().run(
#     subject_ids=['work_sub_dir'],
#     visit_ids=['work_session_dir'], work_dir=WORK_PATH)

study = cls(
    name='test_mixin',
    project_id='data', archive=LocalArchive(input_dir),
  
    inputs=inputs)
study.plot_mean_displacement_pipeline().run(
    subject_ids=['cache'],
    visit_ids=['STUDYMRIMC_MC'], work_dir=WORK_PATH)
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
